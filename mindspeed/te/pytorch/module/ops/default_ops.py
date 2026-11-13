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

from mindspeed.te.pytorch.fp8 import fp8_matmul, get_matmul_wise_by_tensor_key
from mindspeed.te.pytorch.module.ops.comm_overlap_ops import CommOverlapOps


class DefaultOps(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta, key=None, fp8_enable=False):

        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * fp8_meta.tp_world_size

        total_input = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed._all_gather_base(total_input, input_.contiguous(), group=fp8_meta.tp_group, async_op=False)

        if not fp8_enable:
            transpose = get_matmul_wise_by_tensor_key(total_input, key)
            output = torch.matmul(
                total_input.t() if transpose[0] else total_input,
                weight.t() if transpose[1] else weight
            )
            return output, total_input, None
        else:
            output, input_fp8, weight_fp8 = fp8_matmul(total_input, weight, fp8_meta, key)
            return output, input_fp8, weight_fp8

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta, key, fp8_enable=False):
        if not fp8_enable:
            transpose = get_matmul_wise_by_tensor_key(input_, key)
            output_ = torch.matmul(
                input_.t() if transpose[0] else input_,
                weight.t() if transpose[1] else weight
            )
        else:
            output_, input_, weight = fp8_matmul(input_, weight, fp8_meta, key)

        dim_size = list(output_.size())
        dim_size[0] = dim_size[0] // fp8_meta.tp_world_size
        output = torch.empty(dim_size, dtype=output_.dtype, device=torch.cuda.current_device())

        torch.distributed._reduce_scatter_base(output, output_.contiguous(), group=fp8_meta.tp_group)
        return output, input_, weight

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta, key=None, fp8_enable=False):

        if not fp8_enable:
            output_ = torch.matmul(input_, weight.t())
        else:
            output_, input_, weight = fp8_matmul(input_, weight, fp8_meta, key)

        if fp8_meta.tp_world_size > 1:
            torch.distributed.all_reduce(output_, group=fp8_meta.tp_group)

        if bias is not None:
            output_ = output_ + bias
        return output_, input_, weight
