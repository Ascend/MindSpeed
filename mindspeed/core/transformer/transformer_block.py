from functools import wraps
import torch
from torch import Tensor

from megatron.core import tensor_parallel, parallel_state, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.training import get_args
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from mindspeed.core.tensor_parallel.comm_autograd_function import auto_grad_sync_gather_along_last_dim, \
    auto_grad_sync_gather_along_first_dim
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPYCollectiveComm


def transformer_block_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.recompute_method != 'block' and not global_args.swap_attention:
            output = forward_func(*args, **kwargs)
        else:
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
        if self.config.fp8:
            from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            )
        else:
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
                hidden_states = checkpoint_handler(custom(l, l + 1))

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
            offset = pipeline_rank * num_layers_per_pipeline_rank
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
