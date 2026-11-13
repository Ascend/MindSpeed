# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl

from mindspeed.core.memory.recompute.activation.activation_recompute_forward import core_activation_recompute_forward_impl


def mindspeed_activation_recompute_forward(self, hidden_states, per_token_scale=None):
    """MLP.
    Core impl, MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    return core_activation_recompute_forward_impl(self, hidden_states, bias_gelu_impl, bias_geglu_impl, get_cuda_rng_tracker, per_token_scale)

