# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.distributed
import torch.distributed as dist
import torch_npu

from mindspeed.core.transformer.moe.moe_feature import parallel_state, tensor_parallel

COMM_STREAM = None


def async_gather_along_first_dim(input_, group=None, output_split_sizes=None, event=None, use_global_buffer=False):
    """Async gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
        async handle.
    """
    global COMM_STREAM
    if group is None:
        group = parallel_state.get_tensor_model_parallel_group()
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())

    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        if use_global_buffer:
            output = parallel_state.get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        if event:
            # multi stream wait event
            if COMM_STREAM is None:
                COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
            with torch_npu.npu.stream(COMM_STREAM):
                event.wait()
                handle = torch.distributed._all_gather_base(output, input_.contiguous(), group=group, async_op=True)
        else:
            handle = torch.distributed._all_gather_base(output, input_.contiguous(), group=group, async_op=True)

    else:
        dim_size[0] = sum(output_split_sizes)
        if use_global_buffer:
            output = parallel_state.get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        if event:
            # multi stream wait event
            if COMM_STREAM is None:
                COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
            with torch_npu.npu.stream(COMM_STREAM):
                event.wait()
                handle = torch.distributed.all_gather(output_tensor_list, input_, group=group, async_op=True)
        else:
            handle = torch.distributed.all_gather(output_tensor_list, input_, group=group, async_op=True)
    return output, handle


class _AsyncAllToAllWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input_, output_split_sizes, input_split_sizes, event=None, async_mark=True):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        handle = None
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        input_ = input_.contiguous()
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
        if event is not None:
            # multi stream wait event
            global COMM_STREAM
            if COMM_STREAM is None:
                COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
            with torch_npu.npu.stream(COMM_STREAM):
                event.wait()
                if async_mark:
                    handle = dist.all_to_all_single(
                        a2a_out,
                        input_.contiguous(),
                        output_split_sizes=output_split_sizes,
                        input_split_sizes=input_split_sizes,
                        group=group,
                        async_op=True
                    )
                else:
                    dist.all_to_all_single(
                        a2a_out,
                        input_.contiguous(),
                        output_split_sizes=output_split_sizes,
                        input_split_sizes=input_split_sizes,
                        group=group,
                    )
        else:
            if async_mark:
                handle = dist.all_to_all_single(
                    a2a_out,
                    input_.contiguous(),
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=True
                )
            else:
                dist.all_to_all_single(
                    a2a_out,
                    input_.contiguous(),
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                )
        return a2a_out, handle

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        grad = grad_output[0]
        return (
            None,
            _AsyncAllToAllWithBackward.apply(ctx.group, grad, ctx.input_split_sizes, ctx.output_split_sizes, None, False)[0],
            None,
            None,
            None,
            None
        )


def async_alltoall_with_backward(group, input_, output_split_sizes_=None, input_split_sizes=None, event=None, async_mark=True):
    """Wrapper for autograd function"""
    return _AsyncAllToAllWithBackward.apply(group, input_, output_split_sizes_, input_split_sizes, event, async_mark)


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
