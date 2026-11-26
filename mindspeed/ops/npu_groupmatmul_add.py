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
import torch_npu

__all__ = ["npu_groupmatmul_add_fp32"]


def npu_groupmatmul_add_fp32(x, dy, grouplist, grad, *, group_list_type=0):
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")
    if isinstance(grouplist, list):
        grouplist = torch.tensor(grouplist, dtype=torch.int64, device=x.device)
    else:
        grouplist = grouplist.to(x.device)
    if group_list_type == 1:
        grouplist = torch.cumsum(grouplist, dim=0)
    grouped_grad = grad.view(grouplist.shape[0], x.shape[-1], dy.shape[-1])
    torch_npu.npu_grouped_matmul_add_(
        grouped_grad,
        x,
        dy,
        grouplist,
        group_list_type=0,
    )
