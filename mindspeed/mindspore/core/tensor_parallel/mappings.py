# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
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

import mindspore # this is the import patch

import torch


def all_to_all_forward(ctx, group, input, output_split_sizes, input_split_sizes): # this a static method
    ctx.group = group
    ctx.output_split_sizes = output_split_sizes
    ctx.input_split_sizes = input_split_sizes

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input

    input = input.contiguous()
    if output_split_sizes is None:
        # Equal split (all2all)
        output = torch.empty_like(input)
    else:
        # Unequal split (all2all-v)
        output = input.new_empty(
            size=[int(sum(output_split_sizes))] + list(input.size()[1:]),
            dtype=input.dtype,
        )
    mindspore.mint.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes.tolist() if output_split_sizes is not None else None,
        input_split_sizes=input_split_sizes.tolist() if output_split_sizes is not None else None,
        group=group._name,)
    return output