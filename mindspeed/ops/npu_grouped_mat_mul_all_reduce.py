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

from typing import Optional, List, Dict
import torch
import torch_npu
from mindspeed.op_builder import GroupedMatMulAllReduceOpBuilder

__all__ = ["npu_grouped_mat_mul_all_reduce"]


op_builder = GroupedMatMulAllReduceOpBuilder()


def npu_grouped_mat_mul_all_reduce(x: List[torch.Tensor],
                                   weight: List[torch.Tensor],
                                   hcomm: str,
                                   *,
                                   bias: Optional[List[torch.Tensor]] = None,
                                   group_list: Optional[List[int]] = None,
                                   split_item: Optional[int] = 0,
                                   reduce_op: str = "sum",
                                   comm_turn: int = 0) -> List[torch.Tensor]:
    if torch.__version__ < '2.1.0':
        if bias is None:
            bias = []
    outputs = op_builder.load().npu_grouped_mat_mul_all_reduce(
                        x, weight, bias, group_list, split_item,
                        hcomm, reduce_op, comm_turn)
    return outputs