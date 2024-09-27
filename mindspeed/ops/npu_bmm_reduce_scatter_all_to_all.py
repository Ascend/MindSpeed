# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu
from torch.library import impl
from mindspeed.op_builder.npu_bmm_reduce_scatter_all_to_all_builder import BatchMatMulReduceScatterAlltoAllOpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY

__all__ = ["npu_bmm_reducescatter_alltoall"]


mindspeed_ops_builder = BatchMatMulReduceScatterAlltoAllOpBuilder()


@impl(AS_LIBRARY, "npu_bmm_reducescatter_alltoall", "PrivateUse1")
def npu_bmm_reducescatter_alltoall_single(x,
                                          weight,
                                          group_ep,
                                          group_ep_worldsize,
                                          group_tp,
                                          group_tp_worldsize,
                                          *,
                                          bias=None,
                                          shard_type=0):
    if x is None:
        raise AssertionError('x must not be None.')
    if weight is None:
        raise AssertionError('weight must not be None.')
    mindspeed_ops = mindspeed_ops_builder.load()
    y = mindspeed_ops.npu_bmm_reducescatter_alltoall(x,
                                                     weight,
                                                     bias,
                                                     group_ep,
                                                     group_ep_worldsize,
                                                     group_tp,
                                                     group_tp_worldsize,
                                                     shard_type)
    return y


def npu_bmm_reducescatter_alltoall(*args, **kwargs):
    return torch.ops.mindspeed.npu_bmm_reducescatter_alltoall(*args, **kwargs)