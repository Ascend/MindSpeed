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

import torch

from .min_comm_cfg import min_comm_config
from .coc_utils import set_context, reshape_to_2D, is_grad_needed


class RewriteColumnAllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        output_parallel = torch.matmul(input_, weight.t())
        if bias is not None:
            output_parallel = output_parallel + bias
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)

        grad_input = grad_output.matmul(weight)
        handle = torch.distributed.all_reduce(grad_input, group=min_comm_config.tp_group, async_op=True)
        grad_weight, grad_bias = None, None

        if is_grad_weight_needed:
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(input_))
            handle.wait()
            grad_bias = grad_output.sum(dim=0) if is_grad_bias_needed and ctx.use_bias else None
        else:
            handle.wait()

        return grad_input, grad_weight, grad_bias


class RewriteRowAllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        output_ = torch.matmul(input_, weight.t())
        output_parallel = min_comm_config.all_reduce(output_)
        if bias is not None:
            output_parallel = output_parallel + bias
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)

        grad_input = grad_output.matmul(weight)
        grad_weight, grad_bias = None, None

        if is_grad_weight_needed:
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(total_input))
            grad_bias = grad_output.sum(dim=0) if is_grad_bias_needed and ctx.use_bias else None

        return grad_input, grad_weight, grad_bias
