# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps

import torch
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors

from megatron.training import get_args
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding as MegatronRotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig as MegatronTransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.pipeline_parallel.schedules import (
    forward_step, 
    backward_step, 
    deallocate_output_tensor,
    check_first_val_step,
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute
)
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.transformer.attention import Attention as MegatronAttention
from megatron.core.transformer import build_module

from mindspeed.core.tensor_parallel.comm_autograd_function import (
    auto_grad_scatter_along_first_dim_then_last_dim,
)
from mindspeed.core.tensor_parallel.comm_autograd_function import (
    auto_grad_sync_gather_along_last_dim,
    auto_grad_sync_gather_along_first_dim
)
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPXCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.embeddings_2d import RotaryEmbedding as RotaryEmbeddingImpl
from mindspeed.core.tensor_parallel.tp_2d.mlp_2d import mlp_init_2d
from mindspeed.core.tensor_parallel.tp_2d.transformer_config import transformer_config_post_init_impl
from mindspeed.core.tensor_parallel.tp_2d.schedules_2d import forward_backward_pipelining_with_interleaving_tp2d
from mindspeed.core.tensor_parallel.tp_2d.self_attention_2d import attention_init_impl
from mindspeed.core.tensor_parallel.tp_2d.self_attention_2d import self_attention_2d_init as self_attention_init_impl
from mindspeed.core.tensor_parallel.tp_2d.parallel_state_2d import initialize_model_parallel_impl


def mindspeed_self_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        config = args[0] if len(args) > 1 else kwargs['config']
        if config.overlap_param_gather:
            config.reset_attention_order = True
        fn(self, *args, **kwargs)
        self_attention_init_impl(self, *args, _initialize_affine_weight_gpu=_initialize_affine_weight_gpu, **kwargs)
    return wrapper


# coupling with cp
def mindspeed_attention_init(
    self,
    config,
    submodules,
    layer_number,
    attn_mask_type,
    attention_type,
    cp_comm_type=None,
    ):
    super(MegatronAttention, self).__init__(config=config)
    return attention_init_impl(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type,
        attention_type,
        cp_comm_type=cp_comm_type,
        parallel_state=parallel_state,
        build_module_func=build_module
    )


# coupling with cp
def mindspeed_initialize_model_parallel_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        initialize_model_parallel_impl(*args, **kwargs, config=get_args())  # megatron api has no `config` param
    return wrapper


# forward method coupling with cp
class MindSpeedRotaryEmbedding2D(RotaryEmbeddingImpl, MegatronRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        MegatronRotaryEmbedding.__init__(self, *args, **kwargs)
        RotaryEmbeddingImpl.__init__(self, *args, config=get_args(), **kwargs)  # megatron api has no `config` param，no self.config

    def get_rotary_seq_len(self, *args, **kwargs):
        rotary_seq_len = MegatronRotaryEmbedding.get_rotary_seq_len(self, *args, **kwargs)
        if self.config.tp_2d:
            rotary_seq_len *= self.config.tp_x
        return rotary_seq_len


def mindspeed_mlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        mlp_init_2d(self, *args, _initialize_affine_weight_gpu=_initialize_affine_weight_gpu, **kwargs)
    return wrapper


def mindspeed_transformer_config_post_init(self):
    super(MegatronTransformerConfig, self).__post_init__()
    return transformer_config_post_init_impl(
        self,
        args=get_args(),
        init_method_normal=init_method_normal,
        scaled_init_method_normal=scaled_init_method_normal
    )


def mindspeed_forward_backward_pipelining_with_interleaving_tp2d(*args, **kwargs):
    return forward_backward_pipelining_with_interleaving_tp2d(
        *args,
        parallel_state=parallel_state,
        p2p_communication=p2p_communication,
        forward_step=forward_step, 
        backward_step=backward_step, 
        model_type_enums=ModelType,
        get_model_config=get_model_config, 
        get_model_type=get_model_type, 
        deallocate_output_tensor=deallocate_output_tensor, 
        check_first_val_step=check_first_val_step, 
        clear_embedding_activation_buffer=clear_embedding_activation_buffer, 
        finish_embedding_wgrad_compute=finish_embedding_wgrad_compute,
        **kwargs
    )


def mindspeed_transformer_block_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        hidden_states = fn(self, *args, **kwargs)
        if self.config.tp_2d and parallel_state.is_pipeline_last_stage():
            hidden_states = auto_grad_sync_gather_along_first_dim(hidden_states, TPXCollectiveComm)
            hidden_states = auto_grad_sync_gather_along_last_dim(hidden_states, TPYCollectiveComm)
        return hidden_states
    return wrapper


def mindspeed_get_tensor_shapes_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # [s, b, h]
        tensor_shapes = fn(*args, **kwargs)
        # `config` is 5th arg in megatron api
        config = kwargs['config'] if 'config' in kwargs else args[5]
        if config.tp_2d:
            tensor_shapes = [[tensor_shape[0] // config.tp_x, tensor_shape[1], tensor_shape[2] // config.tp_y]
                             for tensor_shape in tensor_shapes]
        return tensor_shapes
    return wrapper


def mindspeed_language_model_embedding_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        encoder_input = fn(self, *args, **kwargs)
        if self.config.tp_2d:
            encoder_input = auto_grad_scatter_along_first_dim_then_last_dim(
                encoder_input, TPXCollectiveComm, TPYCollectiveComm
            )
        return encoder_input
    return wrapper


def mindspeed_allreduce_layernorm_grads_wrapper(fn):
    @wraps(fn)
    def wrapper(model, config):
        fn(model, config)
        layer_norm_2d_grads = []
        for model_chunk in model:
            for _, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
                if param.requires_grad and getattr(param, "2d_tp", False):
                    layer_norm_2d_grad = param.main_grad
                    layer_norm_2d_grads.append(layer_norm_2d_grad.data)
        if layer_norm_2d_grads:
            coalesced = _flatten_dense_tensors(layer_norm_2d_grads)
            torch.distributed.all_reduce(coalesced, group=TPXCollectiveComm.get_comm_group())
            for buf, synced in zip(
                layer_norm_2d_grads, _unflatten_dense_tensors(coalesced, layer_norm_2d_grads)
            ):
                buf.copy_(synced)
    return wrapper
