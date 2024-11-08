#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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
import threading

import torch
import torch_npu
import torch.distributed as dist

from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_expert_model_parallel_group
)
from megatron.training import get_args
from mindspeed.core.weight_grad_store import WeightGradStore


class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AsyncCommUtilsDataSingleton(metaclass=SingletonMeta):
    def __init__(self):
        self.all2all_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        self.tp_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        self.fw_rs_output_ampipe = []
        self.fw_rs_event_ampipe = []
        self.fw_ar_output_ampipe = []
        self.fw_ar_event_ampipe = []
        self.fw_ag_output = []


def get_async_comm_utils_data_instance():
    return AsyncCommUtilsDataSingleton()


def get_fw_ag_output():
    return get_async_comm_utils_data_instance().fw_ag_output


def get_fw_ar_rs_output_ampipe(sequence_parallel):
    if sequence_parallel:
        output_list = get_async_comm_utils_data_instance().fw_rs_output_ampipe
        event_list = get_async_comm_utils_data_instance().fw_rs_event_ampipe
    else:
        output_list = get_async_comm_utils_data_instance().fw_ar_output_ampipe
        event_list = get_async_comm_utils_data_instance().fw_ar_event_ampipe

    if not output_list or not event_list:
        return None

    handle = event_list.pop(0)
    handle.wait()
    return output_list.pop(0)


def async_fw_all_reduce_scatter_ampipe(input_, sequence_parallel):
    world_size = get_tensor_model_parallel_world_size()
    if sequence_parallel:
        # reduce scatter
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] // world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        handle = torch.distributed._reduce_scatter_base(
            output, input_.contiguous(), group=get_tensor_model_parallel_group(), async_op=True
        )
        get_async_comm_utils_data_instance().fw_rs_output_ampipe.append(output)
        get_async_comm_utils_data_instance().fw_rs_event_ampipe.append(handle)
    else:
        # all reduce
        handle = torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group(), async_op=True)
        get_async_comm_utils_data_instance().fw_ar_output_ampipe.append(input_)
        get_async_comm_utils_data_instance().fw_ar_event_ampipe.append(handle)


def async_all_gather(input_, a2a_event=None, is_use_global_memory_buffer=False, is_bwd=False, is_save_input=False):
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input_.size())
    new_dim_size = dim_size[0] * world_size
    dim_size[0] = new_dim_size
    if is_bwd:
        is_save_input = True

    if is_use_global_memory_buffer:
        ag_out = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        ag_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    input_ = input_.contiguous()
    if a2a_event:
        # multi stream wait event
        if get_async_comm_utils_data_instance().tp_stream is None:
            get_async_comm_utils_data_instance().tp_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(get_async_comm_utils_data_instance().tp_stream):
            a2a_event.wait()
            if is_save_input and get_args().use_nanopipe and WeightGradStore.is_decoupleBlock:
                WeightGradStore.save_grad_output(input_.clone().detach())
            handle = torch.distributed._all_gather_base(
                ag_out, input_, group=get_tensor_model_parallel_group(), async_op=True
            )
    else:
        handle = torch.distributed._all_gather_base(
            ag_out, input_, group=get_tensor_model_parallel_group(), async_op=True
        )
    if is_bwd:
        return ag_out, handle
    return input_, ag_out, handle


def async_fw_all_gather(input_, a2a_event=None, is_use_global_memory_buffer=False,
                        ampipe_with_mlp_multistream=False, index=0):
    input_, ag_out, handle = async_all_gather(input_, a2a_event, is_use_global_memory_buffer)
    if ampipe_with_mlp_multistream:
        get_async_comm_utils_data_instance().fw_ag_output[index] = ag_out
    else:
        get_async_comm_utils_data_instance().fw_ag_output.append(ag_out)
    return input_, handle


def async_all_to_all(input_, event=None):
    input_ = input_.contiguous()
    output = torch.empty_like(input_)
    if event:
        # multi stream wait event
        if get_async_comm_utils_data_instance().all2all_stream is None:
            get_async_comm_utils_data_instance().all2all_stream = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(get_async_comm_utils_data_instance().all2all_stream):
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
