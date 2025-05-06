# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.distributed
import torch.distributed as dist
import torch_npu

from mindspeed.core.transformer.moe.moe_feature import parallel_state

COMM_STREAM = None


def async_all_gather(input_, group, event=None, is_use_get_global_memory_buffer=False):
    world_size = torch.distributed.get_world_size(group)
    dim_size = list(input_.size())
    new_dim_size = dim_size[0] * world_size
    dim_size[0] = new_dim_size

    if is_use_get_global_memory_buffer:
        ag_out = parallel_state.get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        ag_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = torch.distributed._all_gather_base(
                ag_out, input_.contiguous(), group=group, async_op=True
            )
    else:
        handle = torch.distributed._all_gather_base(
            ag_out, input_.contiguous(), group=group, async_op=True
        )
    return input_, ag_out, handle


def async_reduce_scatter(input_, group, event=None, stream=None, is_use_get_global_memory_buffer=False):
    world_size = dist.get_world_size(group)
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] // world_size

    if is_use_get_global_memory_buffer:
        rs_out = parallel_state.get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        rs_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    if event or stream:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            if event:
                event.wait()
            if stream:
                torch.cuda.current_stream().wait_stream(stream)
            handle = torch.distributed._reduce_scatter_base(
                rs_out, input_.contiguous(), group=group, async_op=True
            )
    else:
        handle = torch.distributed._reduce_scatter_base(
            rs_out, input_.contiguous(), group=group, async_op=True
        )
    return input_, rs_out, handle


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.cuda.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True
            )
    else:
        # use handle to control comm
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True
        )
    return input_, a2a_out, handle
