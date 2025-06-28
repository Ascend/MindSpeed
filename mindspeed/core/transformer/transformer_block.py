# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps
from contextlib import nullcontext
import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, parallel_state, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from mindspeed.core.tensor_parallel.comm_autograd_function import auto_grad_sync_gather_along_last_dim, \
    auto_grad_sync_gather_along_first_dim
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPYCollectiveComm


def transformer_block_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        output = transformer_block_checkpointed_forward(*args, **kwargs)
        return output

    return row_parallel_forward


def transformer_block_checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
):
    """Forward method with activation checkpointing."""

    def custom(start: int, end: int):
        def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
        ):
            for index in range(start, end):
                layer = self._get_layer(index)
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=None,
                    packed_seq_params=packed_seq_params,
                )
            return hidden_states, context

        return custom_forward

    def checkpoint_handler(forward_func):
        return tensor_parallel.checkpoint(
            forward_func,
            self.config.distribute_saved_activations,
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
        )

    # Checkpoint the input activation of only a set number of individual
    # Transformer layers and skip the rest.
    # A method fully use the device memory removing redundant re-computation.
    global_args = get_args()
    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and
        # checkpoint the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        if not global_args.swap_attention:
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(l, l + self.config.recompute_num_layers)
                )

                l += self.config.recompute_num_layers
        else:
            for l in range(self.num_layers_per_pipeline_rank):
                hidden_states, context = custom(l, l + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
    elif self.config.recompute_method == 'block':
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        vpp_size = self.config.virtual_pipeline_model_parallel_size
        if vpp_rank is None or not global_args.enable_recompute_layers_per_pp_rank:
            vpp_rank = 0
        if vpp_size is None or not global_args.enable_recompute_layers_per_pp_rank:
            vpp_size = 1
        for l in range(self.num_layers_per_pipeline_rank):
            # The number of layers each pipeline rank recomputes is self.recompute_num_layers.
            # If self.recompute_num_layers cannot divide exactly  the number of layers in each pp rank,
            # we try to balance the number of recomputed layers in each model chunk.
            # e.g. with 8 layers, 2 stages, and 2 virtual stages, the assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]   [4, 5]
            # Stage 1: [2, 3]   [6, 7]
            # With self.recompute_num_layers = 2, we will recompute layers 0,4 for stage 0, and 2,6 for stage 1.
            # With self.recompute_num_layers = 3, we will recompute layers 0,1,4 for stage 0, and 2,3,6 for stage 1.
            def should_recompute():
                if global_args.reduce_recompute_for_last_chunk:
                    def is_last_layer():
                        return (l == self.num_layers_per_pipeline_rank - 1) and mpu.is_pipeline_last_stage()

                    return ((l * vpp_size + vpp_rank) < self.config.recompute_num_layers) and not is_last_layer()
                else:
                    return (l * vpp_size + vpp_rank) < self.config.recompute_num_layers

            if should_recompute() and not global_args.swap_attention:
                hidden_states, context = checkpoint_handler(custom(l, l + 1))
            else:
                hidden_states, context = custom(l, l + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

    return hidden_states


class NoopTransformerLayer(MegatronModule):
    def __init__(self, layer_number):
        super().__init__(None)
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, context, context_mask, rotary_pos_emb, inference_params, packed_seq_params):
        return hidden_states.clone(), context


def _get_layer_offset(args):
    num_layers = args.num_layers
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

    num_layers_per_pipeline_rank = (
        num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )
    if args.schedules_method == 'dualpipev':
        num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        total_num_layers = num_layers
        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
        total_virtual_chunks = total_num_layers // vp_size
        offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

    else:
        # Each stage gets a contiguous set of layers.
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if getattr(args, 'dualpipev_first_chunk', True):
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = num_layers - (pipeline_rank + 1) * num_layers_per_pipeline_rank

        else:
            offset = 0
    return offset


def _build_layers(self):
    args = get_args()

    def build_layer(layer_spec, layer_number):
        global_layer_number = _get_layer_offset(args) + layer_number
        if (hasattr(args, 'noop_layers') and isinstance(args.noop_layers, set)
                and global_layer_number - 1 in args.noop_layers):
            return NoopTransformerLayer(global_layer_number)
        return build_module(layer_spec, config=self.config, layer_number=layer_number,)

    self.layers = torch.nn.ModuleList(
        [
            build_layer(layer_spec, i + 1)
            for i, layer_spec in enumerate(self.submodules.layer_specs)
        ]
    )

    if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
    else:
        self.final_layernorm = None  # Either this or nn.Identity


def transformer_block_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        hidden_states = fn(*args, **kwargs)
        if get_args().tp_2d and parallel_state.is_pipeline_last_stage():
            hidden_states = auto_grad_sync_gather_along_first_dim(hidden_states, TPXCollectiveComm)
            hidden_states = auto_grad_sync_gather_along_last_dim(hidden_states, TPYCollectiveComm)
        return hidden_states
    return wrapper


def transformer_block_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
):
    # hidden_states (float): [s, b, h]
    # attention_mask (bool): [1, 1, s, s]

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
        inp=hidden_states, requires_grad=True, keep_graph=True,
    )

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        args = get_args()
        import transformer_engine  # To keep out TE dependency when not training in fp8
        from mindspeed.te.fp8.metadata import FP8Config
        from mindspeed.te.fp8.fp8 import fp8_autocast
        from mindspeed.te.fp8.recipes.recipe import RecipeConfig
        from mindspeed.te.fp8.recipes import SCALING_TYPE_MAP
        from mindspeed.te.fp8.constants import Format
        scale_type = SCALING_TYPE_MAP[args.fp8_recipe]

        if self.config.fp8 == 'hybrid':
            fp8_config = FP8Config(
                inputs=(scale_type, RecipeConfig(fp8_format=Format.E4M3)),
                weight=(scale_type, RecipeConfig(fp8_format=Format.E4M3)),
                grads=(scale_type, RecipeConfig(fp8_format=Format.E5M2)),
                default=(scale_type, RecipeConfig())
            )
        elif self.config.fp8 == 'hif8':
            fp8_config = FP8Config(
                inputs=(scale_type, RecipeConfig(fp8_format=Format.HiF8)),
                weight=(scale_type, RecipeConfig(fp8_format=Format.HiF8)),
                grads=(scale_type, RecipeConfig(fp8_format=Format.HiF8)),
                default=(scale_type, RecipeConfig())
            )
        else:
            fp8_config = FP8Config(default=(scale_type, RecipeConfig()))
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)

        fp8_context = fp8_autocast(enabled=True, fp8_config=fp8_config, fp8_group=fp8_group)
    else:
        fp8_context = nullcontext()

    with rng_context and fp8_context:
        # Forward pass.
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )
        else:
            for layer in self.layers:
                with self.offload_context:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                    )

                if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    # Final layer norm.
    if self.post_process and self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states