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
import torch.nn as nn


class MindSpeedTELayernorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps=1e-5, sequence_parallel=False, zero_centered_gamma=False, **kwargs):
        super(MindSpeedTELayernorm, self).__init__(hidden_size, eps=eps)
        self.sequence_parallel = sequence_parallel
        self.zero_centered_gamma = zero_centered_gamma
        setattr(self.weight, 'sequence_parallel', sequence_parallel)
        setattr(self.bias, 'sequence_parallel', sequence_parallel)
        if self.zero_centered_gamma:
            raise NotImplementedError("Zero-centered gamma is not supported in this dummy implementation.")
