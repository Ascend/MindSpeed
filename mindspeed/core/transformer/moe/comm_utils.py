# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import einops
import torch
import torch.distributed
import torch.distributed as dist
import torch_npu
from megatron.core import parallel_state
from megatron.core.parallel_state import get_global_memory_buffer, get_tensor_model_parallel_rank

from typing import Optional, List

COMM_STREAM = None


def async_all_gather(input_, group, event=None, is_use_get_global_memory_buffer=False, last_dim=False):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
    if last_dim:
        rank = get_tensor_model_parallel_rank()
        ag_out = [torch.empty_like(input_) for _ in range(world_size)]
        ag_out[rank] = input_
    else:
        dim_size = list(input_.size())
        new_dim_size = dim_size[0] * world_size
        dim_size[0] = new_dim_size

        if is_use_get_global_memory_buffer:
            ag_out = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            ag_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            if last_dim:
                handle = torch.distributed.all_gather(ag_out, input_.contiguous(), group=group, async_op=True)
            else:
                handle = torch.distributed._all_gather_base(
                    ag_out, input_.contiguous(), group=group, async_op=True
                )
    else:
        if last_dim:
            handle = torch.distributed.all_gather(ag_out, input_.contiguous(), group=group, async_op=True)
        else:
            handle = torch.distributed._all_gather_base(
                ag_out, input_.contiguous(), group=group, async_op=True
            )
    return input_, ag_out, handle


def async_reduce_scatter(input_, group, event=None, stream=None, is_use_get_global_memory_buffer=False):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] // world_size
    if is_use_get_global_memory_buffer:
        rs_out = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
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
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
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
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True
        )
    return input_, a2a_out, handle


def transfer_tensor_last_dim_to_first(input_x):
    num_dims = input_x.dim()
    return einops.rearrange(input_x, "... lst -> lst ...").contiguous(), num_dims


def transfer_tensor_first_dim_to_last(input_x, num_dims):
    return einops.rearrange(input_x, "first ... -> ... first").contiguous()


def _gather_no_grad(input_: torch.Tensor, output_split_sizes=None, group=None):
    if group is None:
        group = parallel_state.get_tensor_model_parallel_group()
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed._all_gather_base(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def _reduce_scatter_no_grad(input_: torch.Tensor, input_split_sizes=None, group=None):
    if group is None:
        group = parallel_state.get_tensor_model_parallel_group()
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        if dim_size[0] % world_size != 0:
            raise ValueError("First dimension of the tensor should be divisible by tensor parallel size")
        dim_size[0] = dim_size[0] // world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    else:
        rank = torch.distributed.get_rank(group)
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))
        output = torch.empty_like(input_tensor_list[rank])
        torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
    return output


class _Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, output_split_sizes=None, group=None):
        """Forward function."""
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        return _gather_no_grad(input_, output_split_sizes, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        output_split_sizes = ctx.output_split_sizes
        group = ctx.group
        return _reduce_scatter_no_grad(grad_output, output_split_sizes, group), None, None


class _ReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, input_split_sizes=None, group=None):
        """Forward function."""
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        return _reduce_scatter_no_grad(input_, input_split_sizes, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        input_split_sizes = ctx.input_split_sizes
        group = ctx.group
        return _gather_no_grad(grad_output, input_split_sizes, group), None, None


# Backport from latest Megatron-LM
def all_gather(
        input_: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.
        group (torch.distributed.ProcessGroup):
            The process group to perform communication.
            Default: None, will use tensor model parallel group.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    return _Gather.apply(input_, output_split_sizes, group)


def reduce_scatter(
        input_: torch.Tensor,
        input_split_sizes: Optional[List[int]] = None,
        group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
        group (torch.distributed.ProcessGroup):
            The process group to perform communication.
            Default: None, will use tensor model parallel group.
    """
    return _ReduceScatter.apply(input_, input_split_sizes, group)
