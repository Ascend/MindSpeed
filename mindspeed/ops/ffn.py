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
from mindspeed.op_builder.ffn_builder import FFNOpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY

__all__ = ["npu_ffn"]


op_builder = FFNOpBuilder()


@impl(AS_LIBRARY, "npu_ffn", "PrivateUse1")
def _npu_ffn(x,
             weight1,
             weight2,
             activation,
             *,
             expert_tokens=None,
             expert_tokens_index=None,
             bias1=None,
             bias2=None,
             scale=None,
             offset=None,
             deq_scale1=None,
             deq_scale2=None,
             antiquant_scale1=None,
             antiquant_scale2=None,
             antiquant_offset1=None,
             antiquant_offset2=None,
             inner_precise=None,
             output_dtype=None):
    return op_builder.load().npu_ffn(x,
                                   weight1,
                                   weight2,
                                   activation,
                                   expert_tokens,
                                   expert_tokens_index,
                                   bias1,
                                   bias2,
                                   scale,
                                   offset,
                                   deq_scale1,
                                   deq_scale2,
                                   antiquant_scale1,
                                   antiquant_scale2,
                                   antiquant_offset1,
                                   antiquant_offset2,
                                   inner_precise,
                                   output_dtype)


def npu_ffn(*args, **kwargs):
    return torch.ops.mindspeed.npu_ffn(*args, **kwargs)