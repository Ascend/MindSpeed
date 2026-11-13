# Copyright 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
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


def gating(self, input: torch.Tensor):
    """Forward pass of the router gate.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Logits tensor.
    """
    if self.weight.device.type == 'cpu':
        # move weights to GPU
        self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
    # Convert to specified datatype for routing computation if enabled
    router_dtype = input.dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    logits = torch.nn.functional.linear(input.to(router_dtype), self.weight)
    return logits
