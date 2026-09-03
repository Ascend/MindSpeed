# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from dataclasses import dataclass

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from mindspeed.core.memory.recompute.activation.activation_recompute_forward import (
    core_activation_recompute_forward_impl,
)


@dataclass
class ActivationImplementations:
    geglu: callable
    gelu: callable
    quick_gelu: callable
    swiglu: callable
    weighted_quick_geglu: callable
    weighted_swiglu: callable
    have_te: bool


def mindspeed_activation_recompute_forward(self, hidden_states, per_token_scale=None, **kwargs):
    """MLP.
    Core impl, MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    implementations = ActivationImplementations(
        geglu=bias_geglu_impl,
        gelu=bias_gelu_impl,
        quick_gelu=quick_gelu,
        swiglu=bias_swiglu_impl,
        weighted_quick_geglu=weighted_bias_quick_geglu_impl,
        weighted_swiglu=weighted_bias_swiglu_impl,
        have_te=HAVE_TE,
    )
    return core_activation_recompute_forward_impl(
        self,
        hidden_states,
        implementations,
        get_cuda_rng_tracker,
        per_token_scale,
        **kwargs,
    )
