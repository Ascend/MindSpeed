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
from mindspeed.op_builder.npu_mm_all_reduce_add_rms_norm_builder import MatmulAllReduceAddRmsNormOpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY

__all__ = ["npu_mm_all_reduce_add_rms_norm"]


op_builder = MatmulAllReduceAddRmsNormOpBuilder()


@impl(AS_LIBRARY, "npu_mm_all_reduce_add_rms_norm", "PrivateUse1")
def npu_mm_all_reduce_add_rms_norm_single(x1,
                                          x2,
                                          residual,
                                          gamma,
                                          hcom,
                                          reduce_op='sum',
                                          epsilon=1e-06,
                                          bias=None,
                                          antiquant_scale=None,
                                          antiquant_offset=None,
                                          dequant_scale=None,
                                          antiquant_group_size=0,
                                          comm_turn=0):
    if x1 is None:
        raise AssertionError('x1 must not be None.')
    if x2 is None:
        raise AssertionError('x2 must not be None.')
    if residual is None:
        raise AssertionError('residual must not be None.')
    if gamma is None:
        raise AssertionError('gamma must not be None.')
    y, normOut = op_builder.load().npu_mm_all_reduce_add_rms_norm(x1,
                                                           x2,
                                                           residual,
                                                           gamma,
                                                           hcom,
                                                           reduce_op,
                                                           epsilon,
                                                           bias,
                                                           antiquant_scale,
                                                           antiquant_offset,
                                                           dequant_scale,
                                                           antiquant_group_size,
                                                           comm_turn)
    return (y.view(residual.shape), normOut.view(residual.shape))


def npu_mm_all_reduce_add_rms_norm(*args, **kwargs):
    return torch.ops.mindspeed.npu_mm_all_reduce_add_rms_norm(*args, **kwargs)