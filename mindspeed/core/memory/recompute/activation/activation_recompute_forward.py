# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
import torch.nn.functional as F

from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
from mindspeed.core.memory.recompute.recompute_common import CheckpointWithoutOutput

from mindspeed.core.memory.recompute.activation.should_recompute import should_recompute_activation


# pylint: disable=too-many-arguments
def core_activation_recompute_forward_impl(self, hidden_states, bias_gelu_impl, bias_geglu_impl, get_cuda_rng_tracker):
    intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
    self.layer_number = getattr(self, "layer_number", None)
    is_recompute_activation = should_recompute_activation(self.layer_number, self.config)

    def activation_function(*function_args):
        intermediate_parallel, bias_parallel = function_args
        if self.config.bias_activation_fusion:
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

        return intermediate_parallel

    if not is_recompute_activation:
        intermediate_parallel = activation_function(intermediate_parallel, bias_parallel)
        output, output_bias = self.linear_fc2(intermediate_parallel)
    else:
        self.activation_checkpoint_manager = CheckpointWithoutOutput(get_cuda_rng_tracker)
        intermediate_parallel = self.activation_checkpoint_manager.checkpoint(activation_function,
                                                                              False,
                                                                              intermediate_parallel,
                                                                              bias_parallel)
        output, output_bias = self.linear_fc2(intermediate_parallel)
        # discard the output of the activation function,
        # which will be restored by recomputation during backward.
        self.activation_checkpoint_manager.discard_output()

        # when backward to output of dense_4h_to_h,
        # recompute and restore the output of activation function.
        if output.requires_grad:
            output.register_hook(self.activation_checkpoint_manager.recompute)

    return output, output_bias
