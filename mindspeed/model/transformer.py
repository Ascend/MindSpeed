# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
import threading
from functools import wraps
from contextlib import nullcontext
from typing import Any
import warnings

import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
from megatron.training import get_args, get_tokenizer
from megatron.core import parallel_state, mpu, tensor_parallel
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import make_viewless_tensor
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.legacy.model.utils import openai_gelu, erf_gelu, get_norm
from megatron.legacy.model.transformer import ParallelMLP, ParallelTransformer, ParallelTransformerLayer, CoreAttention, \
    FlashSelfAttention, ParallelAttention
from megatron.core.enums import ModelType
from megatron.legacy.model.enums import AttnType, AttnMaskType, LayerType
from megatron.legacy.model.transformer import (_get_num_layers,
                                               _get_layer_type,
                                               bias_dropout_add_fused_train,
                                               bias_dropout_add_fused_inference,
                                               get_bias_dropout_add)
from megatron.legacy.model.fused_bias_gelu import bias_gelu_impl
from megatron.core.tensor_parallel.mappings import _split_along_first_dim
from megatron.core.transformer.module import MegatronModule

from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ulysses,
                                           get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.parallel_state import get_tensor_model_parallel_world_size_for_nd1_dim1
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPXOverlapCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPYOverlapCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.moe.ampipe.ampipe import AttMoEPipe
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.tensor_parallel.layers import Nd_ParallelLinear
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton, get_slopes
from mindspeed.moe.ampipe.ampipe_args import ForwardArgs
from mindspeed.moe.utils import (get_slice_indices_from_order_to_disorder,
                                 get_slice_indices_from_disorder_to_order,
                                 all_gather_along_first_dim)
from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info

try:
    from einops import rearrange
except ImportError:
    rearrange = None

_GLOBAL_ATTN_MASK = None


class Alibi:
    _instance = None
    alibi = None
    matmul_result = None
    output_size = None
    lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        else:
            with cls.lock:
                cls._instance = super().__new__(cls)
                return cls._instance


def _get_inverted_mask(attention_mask, alibi):
    inverted_mask = attention_mask.to(alibi.dtype)
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), float("-inf")
    )
    return inverted_mask.to(alibi.device) + alibi.unsqueeze(0)


def _build_alibi_tensor(max_seq_len, num_attention_heads, square_alibi_mask, fill_neg_inf):
    def _fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def _buffered_future_mask(maxpos, alibi, attn_heads):
        _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
        _future_mask = _future_mask.unsqueeze(0) + alibi
        return _future_mask[:attn_heads, :maxpos, :maxpos]

    slopes = torch.Tensor(get_slopes(num_attention_heads))
    if square_alibi_mask:
        position_point = torch.arange(max_seq_len) - max_seq_len + 1
        position_point = position_point.unsqueeze(0).unsqueeze(0).expand(num_attention_heads, max_seq_len, -1)
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    else:
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

    # Select the part of the tensor that corresponds to our tensor parallel index.
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_index = parallel_state.get_tensor_model_parallel_rank()
    alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

    if fill_neg_inf:
        return _buffered_future_mask(max_seq_len, alibi, num_attention_heads)

    return alibi


def core_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)

        args = get_args()
        self.hidden_size_per_partition = self.hidden_size_per_partition // arg[1].context_parallel_size
        self.square_alibi_mask = args.square_alibi_mask
        self.fill_neg_inf = args.fill_neg_inf
        self.beta = 1.0
        self.config = arg[1]
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        if args.position_embedding_type == 'alibi':
            self.alibi = Alibi()
            alibi = _build_alibi_tensor(args.seq_length,
                                        self.config.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if self.config.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif self.config.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi
        else:
            self.alibi = None

    return wrapper


def core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                   query_layer.size(2),
                   query_layer.size(0),
                   key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.reshape(output_size[2],
                                      output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                               output_size[0] * output_size[1], -1)

    if self.alibi is None:
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),
            key_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0, alpha=(1.0 / self.norm_factor))
    else:
        if self.alibi.matmul_result is None or self.alibi.output_size != output_size:
            args = get_args()

            self.alibi.output_size = output_size
            alibi = _build_alibi_tensor(args.seq_length,
                                        self.config.num_attention_heads,
                                        args.square_alibi_mask,
                                        args.fill_neg_inf
                                        ).to(torch.cuda.current_device())
            if self.config.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif self.config.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)
            self.alibi.alibi = alibi

            if self.fill_neg_inf:
                _alibi = self.alibi.alibi[:, :output_size[3], :output_size[3]]
                attention_mask = attention_mask.repeat(output_size[0], 1, 1, 1)[:output_size[0], :, :, :]
                self.alibi.matmul_result = _get_inverted_mask(attention_mask, _alibi).view(-1, output_size[2],
                                                                                           output_size[2]).contiguous()
            else:
                self.alibi.matmul_result = self.alibi.alibi[:, :, :output_size[3]].repeat(output_size[0], 1, 1)

        q_trans = query_layer.transpose(0, 1).contiguous()
        k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
        matmul_result = self.beta * self.alibi.matmul_result + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

        # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    if self.square_alibi_mask:
        attention_scores = torch.max(
            attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
        )
        attention_probs = torch.nn.functional.softmax(attention_scores, -1)
    else:
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                   value_layer.size(2),
                   query_layer.size(0),
                   value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                   output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                           output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
                              (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


class NoopTransformerLayer(MegatronModule):
    def __init__(self, layer_number):
        super().__init__(None)
        self.layer_number = layer_number

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states.clone()


def parallel_transformer_init(self, config,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()

        args = get_args()

        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_norm = post_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl
        self.retro_add_retriever = args.retro_add_retriever

        # Store activation checkpoiting flag.
        self.recompute_granularity = config.recompute_granularity
        self.recompute_method = config.recompute_method
        self.recompute_num_layers = config.recompute_num_layers
        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel

        self.sequence_parallel = config.sequence_parallel

        # Transformer Engine Init.
        self.transformer_engine_v_0_10 = False
        self.transformer_engine_v_0_11 = False
        self.transformer_engine_v_0_8 = False
        if self.transformer_impl == 'transformer_engine':
            global transformer_engine
            import transformer_engine
            from importlib.metadata import version
            from pkg_resources import packaging

            te_version = packaging.version.Version(version("transformer-engine"))
            if te_version >= packaging.version.Version("0.8.0"):
                self.transformer_engine_v_0_8 = True
            if te_version >= packaging.version.Version("0.10.0"):
                self.transformer_engine_v_0_10 = True
            if te_version >= packaging.version.Version("0.11.0"):
                self.transformer_engine_v_0_11 = True

            del version, packaging

            assert not args.squared_relu, "TransformerEngine does not support squared relu activation."

        self.use_fp8 = config.fp8 is not None
        self.fp8_recipe = None
        self.fp8_group = None
        if self.use_fp8:
            assert args.transformer_impl == 'transformer_engine', \
                'transformer-engine required for fp8 training and inference'
            self.fp8_group = mpu.get_amax_reduction_group()
            if config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("The DelayedScaling recipe only supports E4M3 and HYBRID formats.")
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=config.fp8_margin,
                interval=config.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=config.fp8_amax_history_len,
                amax_compute_algo=config.fp8_amax_compute_algo,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )

        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.checkpoint_core_attention = config.recompute_granularity == 'selective'

        # Number of layers.
        self.num_layers = _get_num_layers(args, model_type,
                                          layer_type==LayerType.decoder)

        self.drop_path_rates = [
            rate.item() for rate in
            torch.linspace(0, self.drop_path_rate, config.num_layers)]

        self.retro_layer_numbers = None
        if model_type == ModelType.retro_decoder:
            retro_layer_start = 6 if config.num_layers <= 15 else 9
            self.retro_layer_numbers = \
                np.arange(retro_layer_start, config.num_layers + 1, 3).tolist()
        if model_type == ModelType.retro_encoder:
            self.retro_layer_numbers = [1]

        # Transformer layers.
        if args.retro_add_retriever:
            assert self.recompute_granularity != 'full', \
                "Full recompute not supported for Retro."
            assert args.transformer_impl == 'local', \
                "Transformer engine does not support Retro layers."
        def build_layer(layer_number):
            if args.transformer_impl == 'local':
                if (hasattr(args, 'noop_layers') and isinstance(args.noop_layers, set)
                        and layer_number - 1 in args.noop_layers):
                    return NoopTransformerLayer(layer_number)

                current_layer_type = _get_layer_type(
                    model_type, layer_type, self.retro_layer_numbers,
                    layer_number)
                return ParallelTransformerLayer(
                    config,
                    layer_number,
                    layer_type=current_layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1])
            else:
                # This argument is only available from TE v0.10 onwards.
                extra_transformer_engine_kwargs = {}
                if self.transformer_engine_v_0_8:
                    extra_transformer_engine_kwargs["bias"] = config.add_bias_linear
                if self.transformer_engine_v_0_10:
                    extra_transformer_engine_kwargs["activation"] = "swiglu" if args.swiglu else "gelu"
                if self.transformer_engine_v_0_11:
                    extra_transformer_engine_kwargs["normalization"] = config.normalization
                assert config.attention_softmax_in_fp32, "TransformerEngine only supports softmax compute in FP32."
                assert (
                    (bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and config.fp16) == config.apply_query_key_layer_scaling
                ), "Unsupported config for apply_query_key_layer_scaling in TransformerEngine."
                return transformer_engine.pytorch.TransformerLayer(
                    config.hidden_size,
                    config.ffn_hidden_size,
                    config.num_attention_heads,
                    layernorm_epsilon=config.layernorm_epsilon,
                    hidden_dropout=config.hidden_dropout,
                    attention_dropout=config.attention_dropout,
                    init_method=config.init_method,
                    output_layer_init_method=config.output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=config.kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_group=mpu.get_tensor_model_parallel_group(),
                    get_rng_state_tracker=tensor_parallel.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    sequence_parallel=config.sequence_parallel,
                    params_dtype=config.params_dtype,
                    apply_residual_connection_post_layernorm=config.apply_residual_connection_post_layernorm,
                    output_layernorm=False,
                    layer_type="encoder",
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    set_parallel_mode=True,
                    fuse_qkv_params=True,
                    **extra_transformer_engine_kwargs)

        if config.virtual_pipeline_model_parallel_size is not None:
            assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = config.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

            # Update dropout rate for Retro encoder.
            if model_type == ModelType.retro_encoder:
                for layer in self.layers:
                    if layer.self_attention.use_flash_attn:
                        layer.self_attention.core_attention_flash.dropout_p = \
                            torch.nn.Dropout(args.retro_encoder_attention_dropout)
                    else:
                        layer.self_attention.core_attention.attention_dropout.p =\
                            args.retro_encoder_attention_dropout
                    layer.hidden_dropout = args.retro_encoder_hidden_dropout

        if self.post_process and self.post_norm:
            # Final layer norm before output.
            self.final_norm = get_norm(config)


def parallel_transformer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        def build_layer(model_type, config, layer_number, layer_type=LayerType.encoder, self_attn_mask_type=AttnMaskType.padding):
            current_layer_type = _get_layer_type(
                model_type, layer_type, self.retro_layer_numbers,
                layer_number)
            return ParallelTransformerLayer(
                config,
                layer_number,
                layer_type=current_layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])
        fn(self, *args, **kwargs)

        argument = get_args()
        if argument.automated_pipeline and argument.num_layer_list and argument.virtual_pipeline_model_parallel_size is None:
            start_layer_num = 1
            self.layers = torch.nn.ModuleList()
            for idx, value in enumerate(argument.num_layer_list):
                if parallel_state.get_pipeline_model_parallel_rank() == idx:
                    self.num_layers = value
                    for layer_num in range(start_layer_num, start_layer_num + value):
                        self.layers.append(build_layer(kwargs['model_type'], args[0], layer_num, self_attn_mask_type=kwargs['self_attn_mask_type']))
                start_layer_num += value
            self.layers = torch.nn.ModuleList(self.layers)

            # Update dropout rate for Retro encoder.
            if kwargs['model_type'] == ModelType.retro_encoder:
                for layer in self.layers:
                    if layer.self_attention.use_flash_attn:
                        layer.self_attention.core_attention_flash.dropout_p = \
                            torch.nn.Dropout(argument.retro_encoder_attention_dropout)
                    else:
                        layer.self_attention.core_attention.attention_dropout.p = \
                            argument.retro_encoder_attention_dropout
                    layer.hidden_dropout = argument.retro_encoder_hidden_dropout
    return wrapper


def set_attention_mask(attn_mask):
    global _GLOBAL_ATTN_MASK
    _GLOBAL_ATTN_MASK = attn_mask


def generate_attention_mask(compress, device):
    global _GLOBAL_ATTN_MASK
    args = get_args()
    if not args.use_flash_attn:
        warnings.warn("Flash Attention is highly recommended")
        _GLOBAL_ATTN_MASK = (torch.tril(torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=bool, device=device), diagonal=-(args.pre_tockens + 1)) \
                                + torch.triu(torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=bool, device=device), diagonal=args.next_tockens + 1))
        return

    if compress:
        seq_len = 2048
    else:
        seq_len = args.seq_length
    
    _GLOBAL_ATTN_MASK = torch.triu(
                            torch.ones((seq_len, seq_len), 
                            device=device, dtype=torch.bool), diagonal=1)


def get_attention_mask():
    global _GLOBAL_ATTN_MASK
    if _GLOBAL_ATTN_MASK is not None:
        return _GLOBAL_ATTN_MASK

    args = get_args()
    should_generate_mask = False
    device = 'npu'

    if args.attention_mask_type == 'causal':
        args.sparse_mode = 2
        should_generate_mask = True
        compress = True

    # ampipe开启在同时不开cp时需要生成全量mask，开cp时生成causal mask
    if args.ampipe_degree > 1 and args.context_parallel_size <= 1:
        args.sparse_mode = 0
        should_generate_mask = True
        compress = False

    # EoD 模式 Ring Attention的实现
    # general 为基线方案，causal 为加速方案
    # 如果 cp > 1 且使用了Ring Attention 并行（包括Hybrid并行）。则Mask为动态生成的，不需要额外的Mask
    if args.reset_attention_mask:
        if args.attention_mask_type == 'general':
            args.sparse_mode = 2
            if args.context_parallel_size == 1 or args.context_parallel_algo == 'ulysses_cp_algo':
                should_generate_mask = True
                compress = True
            else:
                args.sparse_mode = 1
                should_generate_mask = False
        else:
            should_generate_mask = True
            compress = True


    if args.attention_mask_on_cpu:
        device = 'cpu'

    if should_generate_mask:
        generate_attention_mask(compress, device)

    return _GLOBAL_ATTN_MASK


def parallel_transformer_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, attention_mask, **kwargs):
        args = get_args()
        if attention_mask is None:
            attention_mask = get_attention_mask()
        return fn(self, hidden_states, attention_mask, **kwargs)
    return wrapper


def parallel_transformer_forward_ampipe(self, hidden_states, attention_mask,
            encoder_output=None, enc_dec_attn_mask=None,
            retriever_input=None,
            retriever_output=None,
            retriever_attn_mask=None,
            inference_params=None,
            rotary_pos_emb=None):
    # hidden_states: [s, b, h]

    # Checks.
    if inference_params:
        assert self.recompute_granularity is None, \
            'inference does not work with activation checkpointing'

    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(
        hidden_states,
        requires_grad=True,
        keep_graph=True,
    )

    # RNG context.
    if self.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    # Forward layers.
    with rng_context:
        # Determine if the current iteration is first microbatch
        if self.num_microbatches_in_previous_step != get_num_microbatches():
            self.microbatch_count = 0 # Reset count on new batch size rampup interval
        self.num_microbatches_in_previous_step = get_num_microbatches()
        is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

        # Forward pass.
        if self.recompute_granularity == 'full':
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask,
                                                       rotary_pos_emb,
                                                       is_first_microbatch)
        else:
            forward_kwargs = {
                'encoder_output': encoder_output,
                'enc_dec_attn_mask': enc_dec_attn_mask,
                'inference_params': inference_params,
            }

            forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
            forward_kwargs['retriever_input'] = retriever_input
            forward_kwargs['retriever_output'] = retriever_output
            forward_kwargs['retriever_attn_mask'] = retriever_attn_mask

            for index in range(self.num_layers):
                layer = self._get_layer(index)

                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    **forward_kwargs)

                # First Retro decoder layer returns both hidden_states
                # and retriever_output. Make retriever_output available
                # to subsequence Retro layers.
                if isinstance(hidden_states, tuple):
                    assert len(hidden_states) == 2
                    hidden_states, retriever_output = hidden_states
                    forward_kwargs["retriever_output"] = retriever_output
        if self.sequence_parallel:
            ampipe_degree = get_args().ampipe_degree
            if ampipe_degree > 1:
                hidden_states = AmpipeLastTransformerLayerRearrange.apply(hidden_states, ampipe_degree)
        # Skip counter update for eval and activation checkpointing
        if torch.is_grad_enabled() and self.training:
            self.microbatch_count += 1

    # Final layer norm.
    if self.post_process and self.post_norm:
        hidden_states = self.final_norm(hidden_states)

    return hidden_states


class AmpipeLastTransformerLayerRearrange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, ampipe_degree) -> Any:
        ag_hidden_states = all_gather_along_first_dim(hidden_states, True)
        hidden_states.untyped_storage().resize_(0)
        seqlen = ag_hidden_states.size(0)

        new_indices = get_slice_indices_from_order_to_disorder(seqlen, ampipe_degree, device=torch.npu.current_device())
        select_hidden_states = torch.index_select(ag_hidden_states, dim=0, index=new_indices)
        hidden_states_chunk = _split_along_first_dim(select_hidden_states)
        hidden_states_chunk = hidden_states_chunk.clone()
        select_hidden_states.untyped_storage().resize_(0)
        ctx.ampipe_degree = ampipe_degree
        return hidden_states_chunk

    @staticmethod
    def backward(ctx, grad_input) -> Any:
        ag_grad_input = all_gather_along_first_dim(grad_input, True)
        grad_input.untyped_storage().resize_(0)
        seqlen = ag_grad_input.size(0)

        new_indices = get_slice_indices_from_disorder_to_order(seqlen, ctx.ampipe_degree, device=torch.npu.current_device())
        select_grad_input = torch.index_select(ag_grad_input, dim=0, index=new_indices)
        grad_output_chunk = _split_along_first_dim(select_grad_input)
        grad_output_chunk = grad_output_chunk.clone()
        select_grad_input.untyped_storage().resize_(0)
        return grad_output_chunk, None


def parallel_mlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.layer_number = None
        _args = get_args()
        if _args.swiglu and _args.use_fused_swiglu:
            self.activation_func = fused_swiglu

        config = args[0]
        is_expert = kwargs.get('is_expert') if 'is_expert' in kwargs.keys() else False

        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2
        if _args.use_nd_matmul:
            self.dense_h_to_4h = Nd_ParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                skip_bias_add=True,
                input_is_parallel=True,
                is_expert=is_expert,
                matmul_id=1
            )
            self.dense_4h_to_h = Nd_ParallelLinear(
                config.ffn_hidden_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=self.add_bias,
                skip_bias_add=True,
                input_is_parallel=True,
                is_expert=is_expert,
                matmul_id=2
            )
        elif _args.tp_2d:
            self.dense_h_to_4h = ParallelLinear2D(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                add_bias=self.add_bias,
                skip_bias_add=True,
                is_expert=is_expert,
                ag_comm_intf=TPXCollectiveComm,
                ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
                rs_comm_intf=TPYCollectiveComm,
                rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
                enable_overlap_ag_with_matmul=False,
                enable_overlap_matmul_with_rs=_args.enable_overlap_matmul_with_rs,
                partition_dim=0,
                enable_backward_overlap_ag_with_matmul=_args.enable_backward_overlap_ag_with_matmul)
            self.dense_4h_to_h = ParallelLinear2D(
                config.ffn_hidden_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                add_bias=self.add_bias,
                skip_bias_add=True,
                ag_comm_intf=TPYCollectiveComm,
                ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
                rs_comm_intf=TPXCollectiveComm,
                rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
                enable_overlap_ag_with_matmul=_args.enable_overlap_ag_with_matmul,
                enable_overlap_matmul_with_rs=False,
                partition_dim=1,
                enable_backward_overlap_ag_with_matmul=_args.enable_backward_overlap_ag_with_matmul)
        else:
            self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                is_expert=is_expert
            )
            self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
                config.ffn_hidden_size,
                config.hidden_size,
                config=config,
                init_method=config.output_layer_init_method,
                bias=self.add_bias,
                skip_bias_add=True,
                input_is_parallel=True,
                is_expert=is_expert
            )
        if _args.use_nanopipe and parallel_state.get_pipeline_model_parallel_world_size() > 1 \
                    and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            setattr(self.dense_h_to_4h, "in_nano", True)
            setattr(self.dense_4h_to_h, "in_nano", True)
        # use dynamic property assignment to ADD pipe_experts attribution
        if not _args.swiglu:
            self.dense_h_to_4h.pipe_experts = _args.use_pipe_experts
            self.dense_4h_to_h.pipe_experts = _args.use_pipe_experts
        if _args.ampipe_degree > 1:
            setattr(self.dense_h_to_4h, "ampipe_degree", _args.ampipe_degree)
            setattr(self.dense_4h_to_h, "ampipe_degree", _args.ampipe_degree)
    return wrapper


def should_recompute(args, layer_number, num_recompute):
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = args.virtual_pipeline_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = args.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = args.num_layers // pp_size
    else:
        layer_per_chunk = args.num_layers

    if vpp_rank is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not args.enable_recompute_layers_per_pp_rank:
        vpp_size = 1
    recompute_priority = ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = args.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True
        else:
            return False

    if num_recompute is None:
        return True
    else:
        return recompute_priority < num_recompute


def should_recompute_activation(layer_number):
    args = get_args()
    if not args.recompute_activation_function or layer_number is None:
        return False

    if args.recompute_in_bubble or args.recompute_in_advance:
        pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(args.virtual_pipeline_model_parallel_size)
        if pipeline_checkpoint_manager.chunk_do_recompute:
            return False
        elif args.recompute_in_bubble:
            return True

    if args.recompute_activation_function_num_layers is not None:
        if args.recompute_activation_function_num_layers < 0:
            raise AssertionError('--recompute-activation-function-num-layers cannot be less than 0.')
        elif args.recompute_activation_function_num_layers > args.num_layers:
            raise AssertionError('--recompute-activation-function-num-layers cannot be greater than the number of layers.')
    return should_recompute(args, layer_number, args.recompute_activation_function_num_layers)


def should_recompute_norm(self):
    args = get_args()
    if not args.recompute_norm or self.layer_number is None:
        return False
    return should_recompute(args, self.layer_number, args.recompute_norm_num_layers)


def parallel_mlp_forward(self, hidden_states):
    self.layer_number = getattr(self, "layer_number", None)
    is_recompute_activation = should_recompute_activation(self.layer_number)
    args = get_args()
    
    def activation_function(*function_args):
        intermediate, bias = function_args

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            intermediate = bias_gelu_impl(intermediate, bias)
        else:
            if bias is not None:
                intermediate = intermediate + bias
            intermediate = self.activation_func(intermediate)
        return intermediate

    if not is_recompute_activation:
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        if not args.use_pipe_experts and args.ampipe_degree > 1 and args.ampipe_tp_sp_comm_overlap:
            from mindspeed.moe.async_comm_utils import get_fw_ar_rs_output_ampipe, async_all_to_all
            last_chunk_output = get_fw_ar_rs_output_ampipe(args.sequence_parallel)
            if last_chunk_output is not None:
                a2a_output, handle = async_all_to_all(last_chunk_output)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        if not args.use_pipe_experts and args.ampipe_degree > 1 and args.ampipe_tp_sp_comm_overlap:
            if last_chunk_output is not None:
                handle.wait()
                return output, output_bias, a2a_output
    else:
        if not args.use_pipe_experts and args.ampipe_degree > 1 and args.ampipe_tp_sp_comm_overlap:
            from mindspeed.moe.async_comm_utils import (get_fw_ar_rs_output_ampipe,
                                                        async_all_to_all)
            last_chunk_output = get_fw_ar_rs_output_ampipe(args.sequence_parallel)
            if last_chunk_output is not None:
                a2a_output, handle = async_all_to_all(last_chunk_output)

        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        self.activation_checkpoint_manager = CheckpointWithoutOutput()
        intermediate_parallel = self.activation_checkpoint_manager.checkpoint(activation_function,
                                                                              False,
                                                                              intermediate_parallel,
                                                                              bias_parallel)
        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)

        # discard the output of the activation function,
        # which will be restored by recomputation during backward.
        self.activation_checkpoint_manager.discard_output()

        # when backward to output of dense_4h_to_h,
        # recompute and restore the output of activation function.
        if output.requires_grad:
            output.register_hook(self.activation_checkpoint_manager.recompute)
        if not args.use_pipe_experts and args.ampipe_degree > 1 and args.ampipe_tp_sp_comm_overlap:
            if last_chunk_output is not None:
                handle.wait()
                return output, output_bias, a2a_output
    return output, output_bias


def flash_self_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        fn(self, *arg, **kwargs)
        args = get_args()

        self.pse = None
        self.pse_type = args.alibi_fusion_attn_type

        if self.pse_type is None:
            self.pse_type = 1  # not use pse
        elif self.pse_type == 0:
            alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(args.seq_length,
                                                                                 args.num_attention_heads,
                                                                                 args.params_dtype,
                                                                                 args.alibi_diagonal_opposite,
                                                                                 1024)
            self.pse = alibi

        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(args.num_attention_heads)

    return wrapper


def flash_self_attention_init_add_config_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *arg, **kwargs):
        if 'config' in kwargs:
            self.config = kwargs.pop('config')
        fn(self, *arg, **kwargs)

    return wrapper


def flash_self_attention_forward(self, q, k, v, attention_mask):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q, k, v: The tensor containing the query, key, and value. (S, B, H, D)
    """
    args = get_args()
    seq_length, _, head_num, head_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

    q, k, v = [rearrange(x, 's b h d -> s b (h d)') for x in [q, k, v]]

    try:
        scale = 1.0 / math.sqrt(head_dim) if self.softmax_scale is None else self.softmax_scale
    except Exception as e:
        raise ValueError('Invalid head_dim: {}'.format(head_dim)) from e
    cp_expanded_by_2d_tp = args.tp_2d and args.tp_y > 1
    if cp_expanded_by_2d_tp:
        tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
    else:
        tp_y_cp_sz = args.context_parallel_size
    if tp_y_cp_sz > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                                         'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
        in_hybrid_mode = False
        if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
            in_hybrid_mode = True

        if not in_hybrid_mode:
            if cp_expanded_by_2d_tp:
                tp_y_cp = TensorParallelYUnionCP()
                cp_group = tp_y_cp.group
                cp_size = tp_y_cp.get_parallel_group_world_size()
                rank = tp_y_cp.get_parallel_rank()
                cp_global_ranks = tp_y_cp.global_ranks
            else:
                cp_group = mpu.get_context_parallel_group()
                cp_size = mpu.get_context_parallel_world_size()
                rank = mpu.get_context_parallel_rank()
                cp_global_ranks = mpu.get_context_parallel_global_ranks()
        else:
            cp_group = get_context_parallel_group_for_hybrid_ring()
            cp_size = get_context_parallel_for_hybrid_ring_world_size()
            rank = get_context_parallel_for_hybrid_ring_rank()
            cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

        cp_para = dict()
        if hasattr(self, 'config'):
            cp_para['megatron_cp_in_bnsd'] = self.config.megatron_cp_in_bnsd
        cp_para['causal'] = args.attention_mask_type == 'causal'
        cp_para['cp_group'] = cp_group
        cp_para['cp_size'] = cp_size
        cp_para['rank'] = rank

        if args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
            cp_para['cp_global_ranks'] = cp_global_ranks
            if args.use_cp_send_recv_overlap:
                if cp_expanded_by_2d_tp:
                    cp_para['cp_group_for_send_recv_overlap'] = tp_y_cp.overlap_group
                else:
                    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap()
            else:
                cp_para['cp_group_for_send_recv_overlap'] = None
            cp_para['pse'] = self.pse
            cp_para['pse_type'] = self.pse_type
            if args.context_parallel_size > 1 and not args.tp_2d:
                cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
                cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
                cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
                cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
            output = ringattn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
        else:
            cp_para['scheduling_info'] = get_scheduling_info()
            output = adaptive_attn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
    else:
        if args.use_fusion_attn_v2:
            output = npu_fusion_attention(
                q, k, v, head_num, args.shape_order,
                pse=self.pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pse_type=self.pse_type,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens,
                keep_prob=1 - self.dropout_p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]
        else:
            output = torch_npu.npu_fusion_attention(
                q, k, v, head_num, args.shape_order,
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.dropout_p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]
    return output


def parallel_attention_init(self, config, layer_number,
                            attention_type=AttnType.self_attn,
                            attn_mask_type=AttnMaskType.padding):
    super(ParallelAttention, self).__init__()
    args = get_args()
    self.layer_number = max(1, layer_number)
    self.attention_type = attention_type
    self.attn_mask_type = attn_mask_type
    self.params_dtype = config.params_dtype
    self.sequence_parallel = config.sequence_parallel
    self.config = config
    self.group_query_attention = args.group_query_attention
    self.num_query_groups = config.num_query_groups

    query_projection_size = config.kv_channels * config.num_attention_heads
    if self.group_query_attention:
        kv_projection_size = config.kv_channels * config.num_query_groups
    else:
        kv_projection_size = config.kv_channels * config.num_attention_heads

    self.use_flash_attn = args.use_flash_attn \
                          and attention_type == AttnType.self_attn \
                          and self.attn_mask_type == AttnMaskType.causal
    if self.use_flash_attn:
        try:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func
        except ImportError:
            try:
                from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
            except ImportError:
                flash_attn_unpadded_func = None
        if flash_attn_unpadded_func is None:
            raise ImportError('FlashAttention is not installed, please install with '
                              'pip install flash-attn')
        assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                      'self-attention for now')
        assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                            'supports causal mask for now')
        if rearrange is None:
            raise ImportError('einops is not installed, please install with pip install einops')

    # Per attention head and per partition values.
    from megatron import core
    self.hidden_size_per_attention_head = core.utils.divide(
        query_projection_size, config.num_attention_heads)

    # Strided linear layer.
    if attention_type == AttnType.self_attn:
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear or config.add_qkv_bias,
            gather_output=False)
    else:
        assert attention_type == AttnType.cross_attn

        if self.group_query_attention:
            raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
        assert query_projection_size == kv_projection_size

        self.query = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)

        self.key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)

    self.core_attention = CoreAttention(self.layer_number, config,
                                        self.attn_mask_type)
    self.checkpoint_core_attention = config.recompute_granularity == 'selective'

    if self.use_flash_attn:
        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=config.attention_dropout, config=config
        )

    # Output.
    self.dense = tensor_parallel.RowParallelLinear(
        query_projection_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=config.add_bias_linear,
        input_is_parallel=True,
        skip_bias_add=True)
    # patch for attention
    patch_for_attention(config, self)


def patch_for_attention(config, self):
    _args = get_args()
    attn_heads_split_num = (
        get_tensor_model_parallel_world_size_for_nd1_dim1()
        if _args.tp_2d
        else mpu.get_tensor_model_parallel_world_size()
    )
    # Per attention head and per partition values.
    self.num_attention_heads_per_partition = config.num_attention_heads // attn_heads_split_num
    if self.group_query_attention:
        if config.num_query_groups % attn_heads_split_num != 0:
            raise NotImplementedError(
                "Currently the num_query_groups should be a multiple of the tensor parallel size"
            )
        self.num_query_groups_per_partition = config.num_query_groups // attn_heads_split_num
    else:
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition
    query_projection_size = config.kv_channels * config.num_attention_heads
    if _args.group_query_attention:
        kv_projection_size = config.kv_channels * config.num_query_groups
    else:
        kv_projection_size = config.kv_channels * config.num_attention_heads
    # qkv bias
    bias = config.add_qkv_bias or config.add_bias_linear
    cp = config.context_parallel_size
    if _args.tp_2d:
        tp_y_cp_sz = cp * _args.tp_y
    else:
        tp_y_cp_sz = cp
    if tp_y_cp_sz > 1 and _args.context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo',
                                                                               'hybrid_adaptive_cp_algo']:
        if _args.tp_2d:
            tp_y_cp = TensorParallelYUnionCP()
            ulysses_group = tp_y_cp.group
        else:
            ulysses_group = mpu.get_context_parallel_group()
        if _args.context_parallel_algo == 'hybrid_cp_algo' or _args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
            ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
        if self.use_flash_attn:
            self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, ulysses_group)
        else:
            self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)
    if _args.use_nd_matmul:
        self.query_key_value = Nd_ParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            skip_bias_add=True,
            input_is_parallel=True,
            matmul_id=1
        )
    elif _args.tp_2d:
        self.query_key_value = ParallelLinear2D(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            add_bias=bias,
            skip_bias_add=True,
            ag_comm_intf=TPXCollectiveComm,
            ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
            rs_comm_intf=TPYCollectiveComm,
            rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
            enable_overlap_ag_with_matmul=False,
            enable_overlap_matmul_with_rs=False,
            partition_dim=0,
            enable_backward_overlap_ag_with_matmul=False)
    else:
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False)
    # dense bias
    bias = _args.add_dense_bias or config.add_bias_linear
    skip_bias_add = _args.skip_bias_add
    # Output.
    if _args.use_nd_matmul:
        self.dense = Nd_ParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            skip_bias_add=True,
            input_is_parallel=True,
            matmul_id=2
        )
    elif _args.tp_2d:
        self.dense = ParallelLinear2D(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            add_bias=bias,
            skip_bias_add=True,
            ag_comm_intf=TPYCollectiveComm,
            ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
            rs_comm_intf=TPXCollectiveComm,
            rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
            enable_overlap_ag_with_matmul=False,
            enable_overlap_matmul_with_rs=False,
            partition_dim=1,
            enable_backward_overlap_ag_with_matmul=_args.enable_backward_overlap_ag_with_matmul)
    else:
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add)
    if _args.use_nanopipe and parallel_state.get_pipeline_model_parallel_world_size() > 1 \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        setattr(self.query_key_value, "in_nano", True)
        setattr(self.dense, "in_nano", True)
    if _args.ampipe_degree > 1:
        setattr(self.query_key_value, 'ampipe_degree', _args.ampipe_degree)
        setattr(self.query_key_value, 'is_dense_h_to_3h', True)


def parallel_attention_forward(self, hidden_states, attention_mask,
            encoder_output=None, inference_params=None,
            rotary_pos_emb=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    is_first_step = False
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)

            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
            is_first_step = True
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================
    if self.attention_type == AttnType.self_attn:

        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            mixed_x_layer,
            [
                (
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
        value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    # duplicate the pos_emb for self attention
    if rotary_pos_emb is not None:
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = ((rotary_pos_emb,) * 2)

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
            :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
            :sequence_end, batch_start:batch_end, ...]


        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb, self.config)
        key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb, self.config)

    if not self.use_flash_attn:
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2)
            value_layer = value_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2)
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
    else:
        if get_args().ampipe_degree > 1:
            return query_layer, key_layer, value_layer
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, attention_mask)

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias


def switch_mlp_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        global_args = get_args()
        if global_args.moe_model_type == 'megatron_moe':
            fn(self, *args, **kwargs)
            return
        from megatron.legacy.model.transformer import SwitchMLP
        super(SwitchMLP, self).__init__()
        config = args[0]
        layer_number = args[1] if len(args) > 1 else None
        from megatron.core.parallel_state import get_expert_model_parallel_group
        from mindspeed.moe.moe import MoE
        from mindspeed.moe.mixtral_parallel_mlpbm import MixtralParallelMLPBM
        try:
            expert_parallel_group = get_expert_model_parallel_group()
        except AttributeError:
            expert_parallel_group = None

        if layer_number is None:
            self.block = MoE(
                config.hidden_size,
                MixtralParallelMLPBM(config, ) if global_args.swiglu else ParallelMLP(config, is_expert=False),
                num_experts=global_args.num_experts,
                ep_size=config.expert_model_parallel_size,
                k=config.moe_router_topk,
                capacity_factor=global_args.moe_train_capacity_factor,
                eval_capacity_factor=global_args.moe_train_capacity_factor,
                aux_loss_coef=config.moe_aux_loss_coeff,
                ep_group=expert_parallel_group,
                noisy_gate_policy=global_args.noisy_gate_policy,
                no_drop=global_args.moe_no_drop,
                dynamic_padding=global_args.moe_dynamic_padding,
                use_sinkhorn=global_args.moe_use_sinkhorn,
                sequence_parallel=config.sequence_parallel
            )
        else:
            if layer_number % global_args.expert_interval == 0:
                self.block = MoE(
                    config.hidden_size,
                    MixtralParallelMLPBM(config, ) if global_args.swiglu else ParallelMLP(config, is_expert=False),
                    num_experts=global_args.num_experts,
                    ep_size=config.expert_model_parallel_size,
                    k=config.moe_router_topk,
                    capacity_factor=global_args.moe_train_capacity_factor,
                    eval_capacity_factor=global_args.moe_train_capacity_factor,
                    aux_loss_coef=config.moe_aux_loss_coeff,
                    ep_group=expert_parallel_group,
                    noisy_gate_policy=global_args.noisy_gate_policy,
                    no_drop=global_args.moe_no_drop,
                    dynamic_padding=global_args.moe_dynamic_padding,
                    use_sinkhorn=global_args.moe_use_sinkhorn,
                    sequence_parallel=config.sequence_parallel
                )
            else:
                self.block = ParallelMLP(config)
        return
    return wrapper


def switch_mlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        global_args = get_args()
        if global_args.moe_model_type == 'megatron_moe':
            return fn(self, *args, **kwargs)
        hidden_states = args[0]
        used_token = args[1] if len(args) > 1 else None
        output = self.block(hidden_states, used_token)
        return output[0], None
    return wrapper


def parallel_transformer_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from megatron.legacy.model.transformer import SwitchMLP
        super(ParallelTransformerLayer, self).__init__()
        global_args = get_args()
        fn(self, *args, **kwargs)
        self.pipe_degree = global_args.ampipe_degree
        self.ampipe_enabled = global_args.ampipe_degree > 1
        if self.mlp.__class__ is SwitchMLP:
            experts_modules = self.mlp.block.moe_layer.experts.experts if global_args.moe_model_type == 'deepspeed_moe' \
                else self.mlp.local_experts
            for expert in experts_modules:
                expert.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number

    return wrapper


def parallel_transformer_layer_forward_ampipe(self, hidden_states, attention_mask,
            encoder_output=None, enc_dec_attn_mask=None,
            retriever_input=None,
            retriever_output=None,
            retriever_attn_mask=None,
            inference_params=None,
            rotary_pos_emb=None):

    # Update the params in case the retro param changes during inference
    # TODO: better redesign with inference param
    args = get_args()
    if args.retro_add_retriever:
        self.retro_num_neighbors = args.retro_num_neighbors
        self.retro_chunk_length = args.retro_chunk_length
        self.retro_retrieved_length = \
            args.retro_num_retrieved_chunks * args.retro_chunk_length

    # hidden_states: [s, b, h]

    # Layer norm at the beginning of the transformer layer.
    norm_output = self.input_norm(hidden_states)

    if self.ampipe_enabled:
        q, k, v = self.self_attention(
            norm_output,
            attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb)
        # release memory to reduce peak memory usage
        del norm_output
        dense_layer = self.self_attention.dense
        ln = self.post_attention_norm
        k, v = [rearrange(x, 's b n d -> s b (n d)') for x in [k, v]]

        ampipe_forward_args = ForwardArgs(
            dense_layer, bias_dropout_add_fused_train, ln, self.mlp.block, self.hidden_dropout
        )
        out_mlp, residual = AttMoEPipe.apply(q, k, v, hidden_states, attention_mask, ampipe_forward_args)

        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_fused_train(
                out_mlp,
                None,
                residual,
                self.hidden_dropout)

        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output

    # Self attention.
    attention_output, attention_bias = \
        self.self_attention(
            norm_output,
            attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb)

    # Residual connection.
    if self.apply_residual_connection_post_norm:
        residual = norm_output
    else:
        residual = hidden_states

    if self.drop_path is None:
        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            norm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)
    else:
        out = torch.nn.functional.dropout(attention_output + attention_bias,
                                          p=self.hidden_dropout,
                                          training=self.training)
        norm_input = residual + self.drop_path(out)

    # Layer norm post the self attention.
    norm_output = self.post_attention_norm(norm_input)

    # Cross attention.
    if self.layer_type == LayerType.encoder:
        pass
    elif self.layer_type == LayerType.decoder:
        norm_input, norm_output = \
            self.default_decoder_cross_attention(
                encoder_output,
                enc_dec_attn_mask,
                norm_input,
                norm_output,
                bias_dropout_add_func)
    elif self.layer_type == LayerType.retro_encoder:
        norm_input, norm_output = \
            self.retro_encoder_cross_attention(
                retriever_output,
                norm_input,
                norm_output,
                bias_dropout_add_func)
    elif self.layer_type in (LayerType.retro_decoder,
                             LayerType.retro_decoder_with_retriever):
        retriever_output, norm_input, norm_output = \
            self.retro_decoder_cross_attention(
                retriever_input,
                retriever_output,
                retriever_attn_mask,
                norm_input,
                norm_output,
                inference_params,
                bias_dropout_add_func)
    else:
        raise Exception("Unsupported layer type, '%s'." %
                        self.layer_type.name)

    # MLP.
    mlp_output, mlp_bias = self.mlp(norm_output)

    # Second residual connection.
    if self.apply_residual_connection_post_norm:
        residual = norm_output
    else:
        residual = norm_input

    if self.drop_path is None:
        if mlp_bias is not None:
            mlp_bias = mlp_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias,
                residual,
                self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

    else:
        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias
        out = torch.nn.functional.dropout(mlp_output,
                                          p=self.hidden_dropout,
                                          training=self.training)
        output = residual + self.drop_path(out)

    if self.layer_type == LayerType.retro_decoder_with_retriever:
        return output, retriever_output
    else:
        return output
