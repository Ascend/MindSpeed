# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
import torch.nn.functional as F

from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.memory.recompute.recompute_common import CheckpointWithoutOutput
from mindspeed.core.memory.recompute.activation import weighted_bias_swiglu_impl
from mindspeed.core.memory.recompute.activation.should_recompute import should_recompute_activation


DENSE_FC1_MEMORY_CONTEXT = '_fb_overlap_dense_fc1_memory_context'


# pylint: disable=too-many-arguments
def core_activation_recompute_forward_impl(
    self, hidden_states, bias_gelu_impl, bias_geglu_impl, get_cuda_rng_tracker, per_token_scale=None
):
    dense_fc1_memory_context = getattr(self, DENSE_FC1_MEMORY_CONTEXT, None)
    dense_fc1_memory_mode = dense_fc1_memory_context.get('mode') if dense_fc1_memory_context is not None else None
    fc1_checkpoint_manager = None
    bias_parallel = None

    if dense_fc1_memory_mode != 'recompute':
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
    else:

        def linear_fc1_forward(fc1_input):
            nonlocal bias_parallel
            intermediate_parallel, bias_parallel = self.linear_fc1(fc1_input)
            return intermediate_parallel

        fc1_checkpoint_manager = CheckpointWithoutOutput(get_cuda_rng_tracker)
        intermediate_parallel = fc1_checkpoint_manager.checkpoint(linear_fc1_forward, False, hidden_states)

    fc1_output = intermediate_parallel
    self.layer_number = getattr(self, "layer_number", None)
    is_recompute_activation = should_recompute_activation(self.layer_number, self.config)

    def activation_function(*function_args):
        intermediate_parallel, bias_parallel, per_token_scale = function_args
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                    else:
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    if bias_parallel is not None:
                        intermediate_parallel = intermediate_parallel + bias_parallel
                    intermediate_parallel = fused_swiglu(intermediate_parallel)
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)

        return intermediate_parallel

    activation_checkpoint_manager = None
    if not is_recompute_activation:
        intermediate_parallel = activation_function(intermediate_parallel, bias_parallel, per_token_scale)
    else:
        activation_checkpoint_manager = CheckpointWithoutOutput(get_cuda_rng_tracker)
        self.activation_checkpoint_manager = activation_checkpoint_manager
        intermediate_parallel = activation_checkpoint_manager.checkpoint(
            activation_function, False, intermediate_parallel, bias_parallel, per_token_scale
        )

    fc1_swap_manager = None
    if dense_fc1_memory_mode == 'swap' and fc1_output.requires_grad:
        swap_manager_factory = dense_fc1_memory_context['swap_manager_factory']
        fc1_swap_manager = swap_manager_factory(fc1_output)
        fc1_swap_manager.async_swap_out(wait_stream=torch.npu.current_stream())
        dense_fc1_memory_context['fc1_swap_manager'] = fc1_swap_manager

    output, output_bias = self.linear_fc2(intermediate_parallel)

    if activation_checkpoint_manager is not None:
        # discard the output of the activation function,
        # which will be restored by recomputation during backward.
        activation_checkpoint_manager.discard_output()

    if fc1_checkpoint_manager is not None and output.requires_grad:
        # The norm output is the FC1 checkpoint input, while the FC1 output is
        # the activation checkpoint input. Restore them in dependency order
        # before FC2 backward consumes the activation output.
        fc1_checkpoint_manager.discard_output()
        norm_checkpoint_manager = dense_fc1_memory_context.get('norm_checkpoint_manager')

        def recompute_dense_fc1(_):
            if norm_checkpoint_manager is not None:
                norm_checkpoint_manager.recompute(None)
            fc1_checkpoint_manager.recompute(None)
            if activation_checkpoint_manager is not None:
                activation_checkpoint_manager.recompute(None)

        output.register_hook(recompute_dense_fc1)
    elif fc1_swap_manager is not None and output.requires_grad:

        def restore_dense_fc1(_):
            fc1_swap_manager.wait_swap_in()
            if activation_checkpoint_manager is not None:
                activation_checkpoint_manager.recompute(None)

        output.register_hook(restore_dense_fc1)
    elif activation_checkpoint_manager is not None and output.requires_grad:
        # When backward reaches FC2, restore the output of the activation function.
        output.register_hook(activation_checkpoint_manager.recompute)

    if per_token_scale is not None:
        assert output_bias is None, "Bias is not supported with per_token_scale"

    return output, output_bias
