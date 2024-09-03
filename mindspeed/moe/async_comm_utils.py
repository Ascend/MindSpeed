# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.distributed as dist

from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_expert_model_parallel_group,
)
from megatron.training import get_args
from megatron.core.parallel_state import is_pipeline_first_stage
from mindspeed.core.weight_grad_store import WeightGradStore


class AsyncCommUtilsData:
    fw_ag_output = []

    all2all_stream = None
    tp_stream = None


if AsyncCommUtilsData.all2all_stream is None and torch.distributed.is_initialized():
    AsyncCommUtilsData.all2all_stream = torch_npu.npu.Stream(device=torch.npu.current_device())

if AsyncCommUtilsData.tp_stream is None and torch.distributed.is_initialized():
    AsyncCommUtilsData.tp_stream = torch_npu.npu.Stream(device=torch.npu.current_device())


def get_fw_ag_output():
    return AsyncCommUtilsData.fw_ag_output


def async_all_gather(input_, event=None, is_use_get_global_memory_buffer=False, is_save_input=False):
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input_.size())
    new_dim_size = dim_size[0] * world_size
    dim_size[0] = new_dim_size

    if is_use_get_global_memory_buffer:
        ag_out = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        ag_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    input_ = input_.contiguous()
    if event:
        # multi stream wait event
        if AsyncCommUtilsData.tp_stream is None:
            AsyncCommUtilsData.tp_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(AsyncCommUtilsData.tp_stream):
            event.wait()
            if is_save_input and get_args().use_nanopipe and WeightGradStore.is_decoupleBlock:
                WeightGradStore.save_grad_output(input_.clone().detach())
            handle = torch.distributed._all_gather_base(
                ag_out, input_, group=get_tensor_model_parallel_group(), async_op=True
            )
    else:
        handle = torch.distributed._all_gather_base(
            ag_out, input_, group=get_tensor_model_parallel_group(), async_op=True
        )
    return input_, ag_out, handle


def async_fw_all_gather(input_, event=None, is_use_get_global_memory_buffer=False):
    input_, ag_out, handle = async_all_gather(input_, event, is_use_get_global_memory_buffer)
    AsyncCommUtilsData.fw_ag_output.append(ag_out)
    return input_, handle


def async_bw_all_gather(input_, event=None, is_use_get_global_memory_buffer=False):
    input_, ag_out, handle = async_all_gather(input_, event, is_use_get_global_memory_buffer, is_save_input=True)
    return ag_out, handle


def async_all_to_all(input_, event=None):
    input_ = input_.contiguous()
    output = torch.empty_like(input_)
    if event:
        # multi stream wait event
        if AsyncCommUtilsData.all2all_stream is None:
            AsyncCommUtilsData.all2all_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(AsyncCommUtilsData.all2all_stream):
            event.wait()
            handle = dist.all_to_all_single(output, input_, group=get_expert_model_parallel_group(), async_op=True)
    else:
        handle = dist.all_to_all_single(output, input_, group=get_expert_model_parallel_group(), async_op=True)
    return output, handle


def async_fw_ar_rs(input_, sequence_parallel):
    world_size = get_tensor_model_parallel_world_size()
    if sequence_parallel:
        # reduce scatter
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] // world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        handle = torch.distributed._reduce_scatter_base(
            output, input_.contiguous(), group=get_tensor_model_parallel_group(), async_op=True
        )
        return output, handle
    else:
        # all reduce
        handle = torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group(), async_op=True)
        return input_, handle
