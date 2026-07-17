# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from dataclasses import dataclass
from megatron.training import get_args
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.extensions.transformer_engine import HAVE_TE
from mindspeed.core.memory.compress_dense.mlp_forward import mlp_forward_impl


@dataclass
class ActImplementations:
    geglu: callable
    gelu: callable
    quick_gelu: callable
    swiglu: callable
    weighted_quick_geglu: callable
    weighted_swiglu: callable
    have_te: bool


def mindspeed_compress_dense_forward(self, hidden_states, per_token_scale=None, **kwargs):
    act_impls = ActImplementations(
        geglu=bias_geglu_impl,
        gelu=bias_gelu_impl,
        quick_gelu=quick_gelu,
        swiglu=bias_swiglu_impl,
        weighted_quick_geglu=weighted_bias_quick_geglu_impl,
        weighted_swiglu=weighted_bias_swiglu_impl,
        have_te=HAVE_TE,
    )
    return mlp_forward_impl(self, hidden_states, act_impls, get_args(), per_token_scale, **kwargs)
