# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd.
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


class CustomSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, split_point):
        ctx.save_for_backward(input_tensor)
        ctx.split_point = split_point

        permuted_local_tokens = input_tensor[:split_point].contiguous()
        permuted_remote_hot_tokens = input_tensor[split_point:].contiguous()

        return permuted_local_tokens, permuted_remote_hot_tokens

    @staticmethod
    def backward(ctx, grad_output_local, grad_output_remote):
        input_tensor, = ctx.saved_tensors
        grad_input = torch.zeros_like(input_tensor)

        split_point = ctx.split_point

        grad_input[:split_point] = grad_output_local
        grad_input[split_point:] = grad_output_remote

        return grad_input, None
