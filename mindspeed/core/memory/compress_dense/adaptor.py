# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from dataclasses import dataclass
from megatron.training import get_args
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from mindspeed.core.memory.compress_dense.mlp_forward import mlp_forward_impl


@dataclass
class ActImplementations:
    geglu: callable
    gelu: callable
    swiglu: callable


def mindspeed_compress_dense_forward(self, hidden_states):
    act_impls = ActImplementations(
        geglu=bias_geglu_impl,
        gelu=bias_gelu_impl,
        swiglu=bias_swiglu_impl
    )
    return mlp_forward_impl(self, hidden_states, act_impls, get_args())