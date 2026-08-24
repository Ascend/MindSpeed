# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from functools import wraps

from torch import Tensor

from megatron.core import tensor_parallel, parallel_state, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.memory.recompute.recompute_common import get_recompute_priority
from mindspeed.core.memory.adaptive_memory.adaptive_memory_swap_manager import SwapManager as AdaptiveMemorySwapManager
from mindspeed.core.memory.adaptive_recomputing.swap_manager import SwapManager as AdaptiveRecomputingSwapManager


def swap_out_by_size(size):
    from megatron.training import get_args

    args = get_args()
    if args.adaptive_memory_optimization:
        return AdaptiveMemorySwapManager().swap_out_by_size(size)
    else:
        return AdaptiveRecomputingSwapManager().swap_out_by_size(size)


def linear_forward_main_grad_wrapper(forward_func):
    @wraps(forward_func)
    def linear_forward_main_grad(
        ctx,
        inputs,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    ):
        output = forward_func(
            ctx,
            inputs,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
        )
        ctx.weight = weight
        return output

    return linear_forward_main_grad


def linear_backward_main_grad_wrapper(backward_func):
    @wraps(backward_func)
    def linear_backward_main_grad(ctx, grad_output):
        class NewCtx:
            pass

        new_ctx = NewCtx()
        inputs, _ = ctx.saved_tensors
        for key in dir(ctx):
            if key == 'saved_tensors':
                setattr(new_ctx, 'saved_tensors', (inputs, ctx.weight))
            elif key.startswith('__') or key == 'saved_variables':
                continue
            else:
                try:
                    getattr(ctx, key)
                except AttributeError:
                    continue
                setattr(new_ctx, key, getattr(ctx, key))
        return backward_func(new_ctx, grad_output)

    return linear_backward_main_grad


def transformer_block_checkpointed_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor,
    context_mask: Tensor,
    rotary_pos_emb: Tensor,
    attention_bias: Tensor,
    packed_seq_params: PackedSeqParams,
    use_inner_fp8_context,
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
            from megatron.core.fp8_utils import get_fp8_context
            from contextlib import nullcontext

            for index in range(start, end):
                layer = self._get_layer(index)
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1) if use_inner_fp8_context else nullcontext()
                )
                with inner_fp8_context:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
            return hidden_states, context

        return custom_forward

    def checkpoint_handler(forward_func):
        if self.config.fp8:
            from megatron.core.extensions.transformer_engine import te_checkpoint

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
    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and
        # checkpoint the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        if not getattr(self.config, 'swap_attention', False):
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states = checkpoint_handler(custom(layer_idx, layer_idx + 1))

                layer_idx += self.config.recompute_num_layers
        else:
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                hidden_states, context = custom(layer_idx, layer_idx + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
    elif self.config.recompute_method == 'block':
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            # The number of layers each pipeline rank recomputes is self.recompute_num_layers.
            # If self.recompute_num_layers cannot divide exactly  the number of layers in each pp rank,
            # we try to balance the number of recomputed layers in each model chunk.
            # e.g. with 8 layers, 2 stages, and 2 virtual stages, the assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]   [4, 5]
            # Stage 1: [2, 3]   [6, 7]
            # With self.recompute_num_layers = 2, we will recompute layers 0,4 for stage 0, and 2,6 for stage 1.
            # With self.recompute_num_layers = 3, we will recompute layers 0,1,4 for stage 0, and 2,3,6 for stage 1.
            layer = self._get_layer(layer_idx)
            recompute_priority = get_recompute_priority(self.config, layer.layer_number)
            should_recompute_layer = recompute_priority < self.config.recompute_num_layers
            if getattr(self.config, 'reduce_recompute_for_last_chunk', False):
                is_last_layer = layer_idx == self.num_layers_per_pipeline_rank - 1 and mpu.is_pipeline_last_stage()
                should_recompute_layer = should_recompute_layer and not is_last_layer

            if should_recompute_layer and not getattr(self.config, 'swap_attention', False):
                hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
            else:
                hidden_states, context = custom(layer_idx, layer_idx + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

    return hidden_states
