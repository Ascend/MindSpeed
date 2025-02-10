# Copied from Megatron-LM: https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from .inner_data_parallel import (
    get_inner_data_parallel_group,
    get_inner_data_parallel_world_size,
    get_inner_data_parallel_rank,
)


def gather_from_inner_dp_region(input_, inner_dp_parallel_output_grad=True):
    return _GatherFromInnerDataParallelRegion.apply(input_, inner_dp_parallel_output_grad)


class _GatherFromInnerDataParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, inner_dp_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, inner_dp_parallel_output_grad=True):
        ctx.inner_dp_parallel_output_grad = inner_dp_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        inner_dp_parallel_output_grad = ctx.inner_dp_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if inner_dp_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_inner_data_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if dim_size[0] % world_size != 0:
        raise ValueError("First dimension of the tensor should be divisible by tensor parallel size")

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_inner_data_parallel_group()
    )
    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_inner_data_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    if dim_size % world_size != 0:
        raise ValueError("First dimension of the tensor should be divisible by tensor parallel size")
    local_dim_size = dim_size // world_size
    rank = get_inner_data_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset: dim_offset + local_dim_size].contiguous()
    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_inner_data_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_inner_data_parallel_group()
    )

    return output


def split_data(data: torch.Tensor, padding_val: int = 0):
    data_num = data.shape[0]
    dp_size = get_inner_data_parallel_world_size()
    data_pad_num = (data_num + dp_size - 1) // dp_size * dp_size
    data_shape = tuple(data.shape[1:])

    # Flatten the data associated with the keys
    flatten_data = data.contiguous().view(-1, *data_shape).cuda()

    if padding_val != 0:
        flatten_data_pad = torch.full((data_pad_num, *data_shape), padding_val, device=torch.cuda.current_device())
    else:
        flatten_data_pad = torch.zeros((data_pad_num, *data_shape), device=torch.cuda.current_device())

    flatten_data_pad[:data_num, ...] = flatten_data
    split_data_list = list(torch.chunk(flatten_data_pad, dp_size, dim=0))
    data = split_data_list[get_inner_data_parallel_rank()]

    return data
