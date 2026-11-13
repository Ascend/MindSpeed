# Copyright 2025 Huawei Technologies Co., Ltd.
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