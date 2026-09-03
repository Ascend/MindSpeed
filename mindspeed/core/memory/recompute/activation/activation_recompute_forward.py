# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
from typing import cast

import torch
import torch.nn.functional as F

from megatron.core.typed_torch import apply_module
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from mindspeed.core.memory.recompute.activation.should_recompute import should_recompute_activation
from mindspeed.core.memory.recompute.recompute_common import CheckpointWithoutOutput


# pylint: disable=too-many-arguments,too-many-branches
def core_activation_recompute_forward_impl(
    self,
    hidden_states,
    implementations,
    get_cuda_rng_tracker,
    per_token_scale=None,
    **kwargs,
):
    del kwargs  # Megatron 0.18 accepts extension kwargs (for example padding_mask) for MLP.forward.

    nvtx_range_push(suffix="linear_fc1")
    intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(hidden_states)
    nvtx_range_pop(suffix="linear_fc1")

    self.layer_number = getattr(self, "layer_number", None)
    is_recompute_activation = should_recompute_activation(self.layer_number, self.config)

    def activation_function(*function_args):
        intermediate_parallel, bias_parallel, per_token_scale = function_args
        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = implementations.weighted_swiglu(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == implementations.quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = implementations.weighted_quick_geglu(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu and quick_gelu with per_token_scale in MLP.")
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = implementations.geglu(intermediate_parallel, bias_parallel)
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = implementations.gelu(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = implementations.swiglu(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and implementations.have_te,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)

        nvtx_range_pop(suffix="activation")
        return intermediate_parallel

    if not is_recompute_activation:
        intermediate_parallel = activation_function(intermediate_parallel, bias_parallel, per_token_scale)
    else:
        self.activation_checkpoint_manager = CheckpointWithoutOutput(get_cuda_rng_tracker)
        intermediate_parallel = self.activation_checkpoint_manager.checkpoint(
            activation_function,
            False,
            intermediate_parallel,
            bias_parallel,
            per_token_scale,
        )

    nvtx_range_push(suffix="linear_fc2")
    output, output_bias = apply_module(self.linear_fc2)(cast(torch.Tensor, intermediate_parallel))
    nvtx_range_pop(suffix="linear_fc2")

    if is_recompute_activation:
        # The activation is consumed by linear_fc2 and can be restored immediately before its backward.
        self.activation_checkpoint_manager.discard_output()
        if output.requires_grad:
            output.register_hook(self.activation_checkpoint_manager.recompute)

    if per_token_scale is not None and output_bias is not None:
        output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
        output_bias = None

    return output, output_bias
