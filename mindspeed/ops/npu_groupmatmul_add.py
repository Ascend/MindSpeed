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
from mindspeed.op_builder import GroupMatmulAddOpBuilder
from mindspeed.ops.npu_matmul_add import NPUVersion, check_npu_version

__all__ = ["npu_groupmatmul_add_fp32"]

groupmatmul_add_op_builder = GroupMatmulAddOpBuilder()


def npu_groupmatmul_add_fp32(x, dy, grouplist, grad):
    if check_npu_version(NPUVersion.A5):
        torch_npu.npu_grouped_matmul_add_(grad.view(grouplist.shape[0], x.shape[-1], dy.shape[-1]), x, dy, grouplist)
    else:
        groupmatmul_add_ops = groupmatmul_add_op_builder.load()
        groupmatmul_add_ops.npu_groupmatmul_add_fp32(x, dy, grouplist.to('npu'), grad)
