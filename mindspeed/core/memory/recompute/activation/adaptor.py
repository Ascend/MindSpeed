# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl

from mindspeed.core.memory.recompute.activation.activation_recompute_forward import core_activation_recompute_forward_impl


def mindspeed_activation_recompute_forward(self, hidden_states):
    """MLP.
    Core impl, MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    return core_activation_recompute_forward_impl(self, hidden_states, bias_gelu_impl, bias_geglu_impl, get_cuda_rng_tracker)

