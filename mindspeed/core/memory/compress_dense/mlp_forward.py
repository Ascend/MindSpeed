# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.nn.functional as F

from mindspeed.core.memory.compress_dense.compress_tensor import ActivationCompress


def mlp_forward_impl(self, hidden_states, act_impls, train_args, per_token_scale=None, **kwargs):
    if kwargs:
        raise TypeError(
            f"compress-dense MLP.forward does not support extra keyword arguments: {', '.join(sorted(kwargs))}"
        )

    bias_geglu_impl = act_impls.geglu
    bias_gelu_impl = act_impls.gelu
    quick_gelu = act_impls.quick_gelu
    bias_swiglu_impl = act_impls.swiglu
    weighted_bias_quick_geglu_impl = act_impls.weighted_quick_geglu
    weighted_bias_swiglu_impl = act_impls.weighted_swiglu
    have_te = act_impls.have_te

    if not hasattr(self, "activation_compress"):
        self.activation_compress = ActivationCompress(
            train_args, "mlp_ctm", [not getattr(self, "self.shared_expert", False)]
        )
    self.activation_compress.compress_and_wait_decompress_async_for_previous_layer(hidden_states)

    intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

    self.activation_compress.decompress_and_wait_compress_async_for_previous_layer(intermediate_parallel)

    if getattr(self.config, "use_te_activation_func", False):
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
                if bias_parallel is not None:
                    raise ValueError(
                        "compress-dense MLP does not support per_token_scale with fused "
                        "SwiGLU when the first linear layer returns a bias."
                    )
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    per_token_scale.unsqueeze(-1),
                    self.config.activation_func_fp8_input_store,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_quick_geglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    per_token_scale.unsqueeze(-1),
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError(
                    "Only support fusion of swiglu and quick_gelu with per_token_scale in compress-dense MLP."
                )
        else:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                    self.config.cpu_offloading and self.config.cpu_offloading_activations and have_te,
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

    self.activation_compress.order_record(intermediate_parallel)

    output, output_bias = self.linear_fc2(intermediate_parallel)

    if per_token_scale is not None and output_bias is not None:
        output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
        output_bias = None

    return output, output_bias
