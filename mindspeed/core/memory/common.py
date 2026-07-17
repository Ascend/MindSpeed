# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from functools import wraps

from torch import Tensor

from megatron.core import tensor_parallel, mpu
from megatron.core.packed_seq_params import PackedSeqParams
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
        tp_group,
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
            tp_group,
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
    use_inner_quantization_context,
    padding_mask=None,
    extract_layer_indices=None,
    layer_offset=0,
):
    """Forward method with activation checkpointing."""
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states = []

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb,
            padding_mask=None,
        ):
            from contextlib import nullcontext
            from megatron.core.fp4_utils import get_fp4_context
            from megatron.core.fp8_utils import get_fp8_context
            from megatron.core.transformer.transformer_layer import TransformerLayer

            for index in range(start, end):
                layer = self.layers[index]
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    else:
                        for key in ('context', 'context_mask', 'attention_bias', 'padding_mask'):
                            layer_kwargs.pop(key, None)
                        hidden_states = layer(**layer_kwargs)
                        context = None
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            return hidden_states, context

        return custom_forward

    def checkpoint_handler(forward_func):
        checkpoint_args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask)
        if self.config.fp8 or self.config.fp4:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                self.pg_collection.tp,
                *checkpoint_args,
            )
        return tensor_parallel.checkpoint(forward_func, self.config.distribute_saved_activations, *checkpoint_args)

    def run_chunk(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        if use_checkpoint:
            hidden_states, context = checkpoint_handler(custom(start, end))
        else:
            hidden_states, context = custom(start, end)(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                padding_mask,
            )
        if self.config.recompute_method == 'uniform':
            if (end - 1 + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)
        elif (start + layer_offset) in extract_layer_indices:
            intermediate_hidden_states.append(hidden_states)

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
                chunk_end = min(
                    layer_idx + self.config.recompute_num_layers,
                    self.num_layers_per_pipeline_rank,
                )
                run_chunk(layer_idx, chunk_end, True)
                layer_idx += self.config.recompute_num_layers
        else:
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                run_chunk(layer_idx, layer_idx + 1, False)
    elif self.config.recompute_method == 'block':
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        vpp_size = self.config.virtual_pipeline_model_parallel_size
        if vpp_rank is None or not getattr(self.config, 'enable_recompute_layers_per_pp_rank', False):
            vpp_rank = 0
        if vpp_size is None or not getattr(self.config, 'enable_recompute_layers_per_pp_rank', False):
            vpp_size = 1
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
            # The closure is called before the loop advances, so it observes this iteration's layer index.
            # pylint: disable=cell-var-from-loop
            def should_recompute():
                if getattr(self.config, 'reduce_recompute_for_last_chunk', False):

                    def is_last_layer():
                        return (layer_idx == self.num_layers_per_pipeline_rank - 1) and mpu.is_pipeline_last_stage()

                    return (
                        (layer_idx * vpp_size + vpp_rank) < self.config.recompute_num_layers
                    ) and not is_last_layer()
                else:
                    return (layer_idx * vpp_size + vpp_rank) < self.config.recompute_num_layers

            # pylint: enable=cell-var-from-loop
            if should_recompute() and not getattr(self.config, 'swap_attention', False):
                run_chunk(layer_idx, layer_idx + 1, True)
            else:
                run_chunk(layer_idx, layer_idx + 1, False)
    else:
        raise ValueError("Invalid activation recompute method.")

    if extract_layer_indices:
        return hidden_states, intermediate_hidden_states
    return hidden_states
