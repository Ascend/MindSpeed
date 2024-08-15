from functools import wraps
import torch
from torch import Tensor

from megatron.core import tensor_parallel, parallel_state, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args


def transformer_block_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.recompute_method != 'block':
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
                packed_seq_params,
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
                packed_seq_params,
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
                packed_seq_params,
            )

    # Checkpoint the input activation of only a set number of individual
    # Transformer layers and skip the rest.
    # A method fully use the device memory removing redundant re-computation.
    global_args = get_args()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = global_args.virtual_pipeline_model_parallel_size
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
                packed_seq_params,
            )

    return hidden_states
