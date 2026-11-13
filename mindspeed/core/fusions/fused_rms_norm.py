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
import torch
import torch_npu


class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False,
                 config=None):
        """RMS Normaliation module

        Args:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.config = config

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def unfused_rmsnorm(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def fused_rmsnorm(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

    def forward(self, x):
        if self.config.use_fused_rmsnorm:
            return self.fused_rmsnorm(x)
        return self.unfused_rmsnorm(x)
