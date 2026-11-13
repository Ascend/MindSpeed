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
import torch_npu

from mindspeed.te.pytorch.fp8 import get_matmul_wise_by_tensor_key
from mindspeed.te.pytorch.fp8.tensor import is_fp8_tensor
from mindspeed.te.pytorch.module.ops import DefaultOps
from mindspeed.te.pytorch.module.ops.comm_overlap_ops import CommOverlapOps
from mindspeed.te.pytorch.module_typing import FP8Metadata, FP8Tensor
from mindspeed.te.pytorch.utils import get_hccl_comm_name


class Mc2Ops(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta, key=None, fp8_enable=False):
        if fp8_enable:
            return DefaultOps.allgather_matmul(input_, weight, bias, fp8_meta, key, fp8_enable)
        hcomm_name = get_hccl_comm_name(fp8_meta.tp_group, fp8_meta.tp_rank)
        transpose = get_matmul_wise_by_tensor_key(input_, key)
        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            x.t() if transpose[0] else x,
            weight.t() if transpose[1] else weight,
            hcomm_name,
            fp8_meta.tp_world_size,
            bias=bias,
            gather_index=0,
        )
        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])
        return output, all_gather_grad_output, None

    @staticmethod
    def fp8_all_gather_matmul(inputs: FP8Tensor, weight: FP8Tensor, bias, fp8_meta: FP8Metadata, key):
        if not is_fp8_tensor(inputs):
            inputs = fp8_meta.quantization(key[0], inputs)
        if not is_fp8_tensor(weight):
            weight = fp8_meta.quantization(key[1], weight)
        output, all_gather_grad_output = inputs.all_gather_matmul(weight, bias, fp8_meta, key)
        return output, all_gather_grad_output, weight

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta, key, fp8_enable=False):
        if fp8_enable:
            return Mc2Ops.fp8_matmul_reduce_scatter(input_, weight, fp8_meta, key, bias)

        hcomm_name = get_hccl_comm_name(fp8_meta.tp_group, fp8_meta.tp_rank)
        transpose = get_matmul_wise_by_tensor_key(input_, key)
        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output = torch_npu.npu_mm_reduce_scatter_base(
            x.T if transpose[0] else x,
            weight.T if transpose[1] else weight,
            hcomm_name, fp8_meta.tp_world_size, reduce_op="sum", bias=bias
        )
        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )
        return output, input_, weight

    @staticmethod
    def fp8_matmul_reduce_scatter(inputs, weight, fp8_meta: FP8Metadata, key, bias):
        if not is_fp8_tensor(inputs):
            inputs = fp8_meta.quantization(key[0], inputs)
        if not is_fp8_tensor(weight):
            weight = fp8_meta.quantization(key[1], weight)
        output = inputs.matmul_reduce_scatter(weight, bias, fp8_meta, key)
        return output, inputs, weight

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta, key=None, fp8_enable=False):
        hcomm_name = get_hccl_comm_name(fp8_meta.tp_group, fp8_meta.tp_rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output = torch_npu.npu_mm_all_reduce_base(
            x, weight.t(), hcomm_name, reduce_op="sum", bias=bias
        )
        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )
        return output, input_, weight
