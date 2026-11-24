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

from functools import reduce

from .coc_parallel_linears_all_reduce import COCRowAllReduceFunction
from .rewrite_parallel_linears_all_reduce import RewriteRowAllReduceFunction
from .coc_utils import get_parallel_num, get_output_shape, allocate_for_output
from .coc_utils import set_context, allocate_for_output, reshape_to_2D, is_grad_needed


class FusedCOCRowAllReduceFunction(RewriteRowAllReduceFunction):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        from mindspeed.ops.lcal_functional import coc_ops

        set_context(ctx, input_, weight, bias)
        parallel_num = get_parallel_num(m=reduce(lambda x, y: x * y, input_.shape[:-1]),
                                        k=weight.shape[1],
                                        n=weight.shape[0])
        if parallel_num == 1:
            return RewriteRowAllReduceFunction.forward(ctx, input_, weight, bias)
        elif parallel_num in [2, 4, 8]:
            return COCRowAllReduceFunction.forward(ctx, input_, weight, bias)

        output_orig_shape = get_output_shape(input_, weight.t(), 1, is_gather=True)
        input_ = reshape_to_2D(input_)
        output_ = allocate_for_output(input_, weight.t(), 1, is_gather=True)

        coc_ops.matmul_all_reduce(input_, weight, output_, bias)
        output_ = output_.reshape(output_orig_shape)

        return output_

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        grad_input = grad_output.matmul(weight)
        grad_weight, grad_bias = None, None
        if is_grad_weight_needed:
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(total_input))
            grad_bias = grad_output.sum(dim=0) if ctx.use_bias and is_grad_bias_needed else None

        return grad_input, grad_weight, grad_bias
