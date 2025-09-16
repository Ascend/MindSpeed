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

from functools import lru_cache
import mindspore
import torch
from torch.ops.comm_func import all_to_all_v_c


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


@lru_cache(maxsize=128)
def _cached_world_size(group):
    return torch.distributed.get_world_size(group=group)


@lru_cache(maxsize=128)
def _cached_myrank(group):
    return torch.distributed.get_rank(group=group)


def all_to_all_forward_a2avc(ctx, group, input, send_count_matrix, mode):  # staticmethod
    ctx.group = group
    ctx.mode = mode

    world_size = _cached_world_size(group=group)
    if world_size == 1:
        return input

    my_rank = _cached_myrank(group=group)
    input_size = input.size()
    in0 = int(input_size[0])
    input = input.contiguous()

    #  build or verify send_count_matrix
    if send_count_matrix is None:
        if ctx.mode == 1:
            if in0 % world_size != 0:
                raise ValueError(f"Uniform distribution of experts requires that input.size(0) should be exactly divided by world_size={world_size} , but the current input.size(0)={in0}")

        chunk = in0 // world_size
        send_count_matrix = [chunk] * (world_size * world_size)
    elif ctx.mode == 1:
        if not isinstance(send_count_matrix, list):
            raise TypeError("send_count_matrix must be a list")
        if len(send_count_matrix) != world_size * world_size:
            raise ValueError(f"send_count_matrix must have length world_size * world_size ({world_size * world_size})")
        
    if ctx.mode == 1:    
        # The total amount sent (the sum of this row) must = the first dimension of the input
        row_sum = sum(send_count_matrix[my_rank * world_size + j] for j in range(world_size))
        if row_sum != in0:
            raise ValueError(f"The total number of tokens to be sent by this rank ({row_sum}) != input.size(0)({in0})")

    # The total received amount (the sum of this column) is used for the  output allocation
    recv_total = sum(send_count_matrix[i * world_size + my_rank] for i in range(world_size))
    output = input.new_empty([int(recv_total)] + list(input_size[1:]), dtype=input.dtype)

    # Save the transposed SCM for backward
    ctx.send_count_matrix_T = [
        send_count_matrix[j * world_size + i]
        for i in range(world_size) 
        for j in range(world_size)
    ]

    if ctx.mode == 1:
        mindspore.communication.comm_func.all_to_all_v_c(
            output, input, send_count_matrix, group=group._name, async_op=False
            )
    elif ctx.mode == 2:
        all_to_all_v_c(
            output, input, send_count_matrix, group=group._name, async_op=False
            )
    else:
        raise ValueError(f" --enable-a2avc should be 1 or 2, but the current value={ctx.mode}")

    return output
