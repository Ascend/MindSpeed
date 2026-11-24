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

import torch_npu

__all__ = ["npu_rotary_position_embedding"]


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    if mode not in (0, 1):
        raise ValueError("mode must be 0 (half) or 1 (interleave)")
    return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="half" if mode == 0 else "interleave")
