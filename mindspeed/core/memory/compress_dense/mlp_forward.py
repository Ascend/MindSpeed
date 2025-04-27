# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import partial
import torch
import torch.nn.functional as F

from mindspeed.core.memory.compress_dense.compress_tensor import ActivationCompress


def mlp_forward_impl(self, hidden_states, act_impls, train_args):
    bias_geglu_impl = act_impls.geglu
    bias_gelu_impl = act_impls.gelu
    bias_swiglu_impl = act_impls.swiglu
    
    if not hasattr(self, "activation_compress"):
        self.activation_compress = ActivationCompress(train_args, "mlp_ctm", [not getattr(self, "self.shared_expert", False)])
    self.activation_compress.compress_and_wait_decompress_async_for_previous_layer(hidden_states)

    intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
    
    self.activation_compress.decompress_and_wait_compress_async_for_previous_layer(intermediate_parallel)
    
    if self.config.bias_activation_fusion:
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
            )
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
    
    self.activation_compress.order_record(intermediate_parallel)
    
    output, output_bias = self.linear_fc2(intermediate_parallel)

    return output, output_bias