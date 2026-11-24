# Copyright 2026 Huawei Technologies Co., Ltd.
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

import os
from functools import wraps

import torch


class FusedLayerNormAffineFunction:
    @staticmethod
    def apply(input_, weight, bias, normalized_shape, eps):
        return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)

    @staticmethod
    def forward(*args, **kwargs):
        return FusedLayerNormAffineFunction.apply(*args, **kwargs)


class FastLayerNormFN:
    @staticmethod
    def apply(input_, weight, bias, eps):
        normalized_shape = torch.Size(weight.numel())
        return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)


def fused_layer_norm_affine(input_, weight, bias, normalized_shape, eps):
    return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)
