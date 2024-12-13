# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional, List

import torch
import torch.distributed as dist


def adjust_tensor_dimensions(tensor, scatter_idx, gather_idx):
    """
    Adjusts the dimensions of a tensor to prepare for an all-to-all operation.

    Args:
        tensor (torch.Tensor): The input tensor.
        scatter_idx (int): The index of the dimension to scatter.
        gather_idx (int): The index of the dimension to gather.
    """
    dims = list(range(tensor.dim()))
    assert scatter_idx != gather_idx
    if gather_idx == 0:
        if scatter_idx != 1:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
        else:
            dims[scatter_idx], dims[gather_idx] = dims[gather_idx], dims[scatter_idx]

    elif gather_idx == 1:
        if scatter_idx != 0:
            # If scatter_idx is not 0, move it to 0
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
    else:
        if scatter_idx == 0:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
        else:
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
    return tensor.permute(dims).contiguous(), dims


def unadjust_tensor_dimensions(tensor, adjusted_dims):
    """
    Reverts the dimensions of a tensor back to their original order after an all-to-all operation.

    Args:
        tensor (torch.Tensor): The tensor with adjusted dimensions.
        adjusted_dims (list): The list of dimensions after adjustment.

    Returns:
        torch.Tensor: The tensor with its dimensions reverted to the original order.
    """
    inverse_dims = [0] * len(adjusted_dims)

    for new_pos, old_pos in enumerate(adjusted_dims):
        inverse_dims[old_pos] = new_pos

    unadjusted_tensor = tensor.permute(inverse_dims).contiguous()
    return unadjusted_tensor


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    scatter_size: Optional[int] = None,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. This function supports non-uniform scatter
    and gather sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        world_size (int): The number of processes in the process group.
        group (dist.ProcessGroup): The process group perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        scatter_size (Optional[int]): The size of the dimension along which the tensor is scattered. Defaults to None.
        gather_size (Optional[int]): The size of the dimension along which the tensor is gathered. Defaults to None.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    input_ = input_.contiguous()
    rank = dist.get_rank(group=group)
    input_shape = list(input_.shape)
    assert 0 <= scatter_dim < len(input_shape), "scatter_dim out of bounds"
    assert 0 <= gather_dim < len(input_shape), "gather_dim out of bounds"
    assert scatter_dim != gather_dim

    adjusted_input, adjusted_input_dims = adjust_tensor_dimensions(input_, scatter_dim, gather_dim)

    scatter_size_per_rank = scatter_size // world_size
    scatter_size_remainder = scatter_size % world_size
    input_split_sizes = [scatter_size_per_rank + (1 if i < scatter_size_remainder else 0) for i in range(world_size)]

    gather_size_per_rank = gather_size // world_size
    gather_size_remainder = gather_size % world_size
    output_split_sizes = [gather_size_per_rank + (1 if i < gather_size_remainder else 0) for i in range(world_size)]

    adjusted_output_dims = adjusted_input_dims
    adjusted_output_dims[1], adjusted_output_dims[0] = adjusted_output_dims[0], adjusted_output_dims[1]
    adjusted_output = torch.empty((gather_size, input_split_sizes[rank], *adjusted_input.shape[2:]),
                                  dtype=adjusted_input.dtype, device=adjusted_input.device)

    dist.all_to_all_single(
        adjusted_output,
        adjusted_input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    output_ = unadjust_tensor_dimensions(adjusted_output, adjusted_output_dims)
    return output_


class _AllToAll(torch.autograd.Function):
    """Custom autograd function that performs an all-to-all communication.
    This function supports non-uniform scatter and gather sizes.
    """
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, scatter_size, gather_size):
        """
        Forward pass: Perform the all-to-all operation by scattering the input tensor along the specified scatter dimension
        and then gathering it along the specified gather dimension.

        Args:
            ctx: The context object to save information for the backward pass.
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            scatter_dim (int): The index of the dimension that needs to be scattered.
            gather_dim (int): The index of the dimension that needs to be gathered.
            scatter_size (int): The size of the scatter dimension.
            gather_size (int): The size of the gather dimension.

        Returns:
            torch.Tensor: The resulting tensor after performing the all-to-all operation.
        """
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.scatter_size = scatter_size
        ctx.gather_dim = gather_dim
        ctx.gather_size = gather_size
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim, scatter_size, gather_size
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Perform the all-to-all operation in reverse by scattering the gradients along the specified
        gather dimension and then gathering them along the specified scatter dimension.

        Args:
            ctx: The context object containing information from the forward pass.
            grad_output (torch.Tensor): The gradient of the output with respect to the loss.

        Returns:
            tuple: The gradient of the input with respect to the loss and `None` for other arguments.
        """
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.gather_size,
            ctx.scatter_size
        )
        return (
            grad_output,
            None,
            None,
            None,
            None,
            None
        )


def _split(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Splits a tensor across the specified dimension and returns the part corresponding to the current rank,
    supporting non-uniform split_sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be split.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to split the tensor. Defaults to -1 (last dimension).
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes, with the remainder
            distributed to the first few processes. Defaults to None.

    Returns:
        torch.Tensor: The part of the tensor corresponding to the current rank in the process group.
    """
    # Ensure split_sizes is a list if provided
    assert split_sizes is None or isinstance(split_sizes, list)

    # skip if only one rank involved
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        return input_

    # Calculate split sizes if not provided
    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size

        # Calculate the size for each process
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)

    # Get the part corresponding to the current rank
    rank = dist.get_rank(pg)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: Optional[List[int]] = None):
    """
    Gathers tensors from all processes in the process group and concatenates them along the specified dimension,
    supporting non-uniform gather_sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be gathered.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to concatenate the gathered tensors. Defaults to -1 (last dimension).
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.

    Returns:
        torch.Tensor: The concatenated tensor after gathering from all processes in the process group.
    """
    # Ensure gather_sizes is a list if provided
    assert gather_sizes is None or isinstance(gather_sizes, list)

    # Skip if only one rank is involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()

    # Prepare the output list with appropriate shapes
    if gather_sizes:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_, dtype=input_.dtype, device=input_.device) for _ in range(world_size)]

    assert input_.device.type == "cuda" or input_.device.type == "npu"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Custom autograd function that gathers the input tensor from all processes in the model parallel region and
    concatenates them.
    During the backward pass, it splits the gradients and scales them according to the specified mode.

    """

    @staticmethod
    def symbolic(graph, input_, process_group, dim, gather_sizes):
        """
        Define the symbolic representation of the custom operation.
        """
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        """
        Forward pass: Gather the input tensor from all processes and concatenate them along the specified dimension.

        Args:
            ctx: The context object to save information for the backward pass.
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            dim (int): The dimension along which to concatenate the gathered tensors.
            gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

        Returns:
            torch.Tensor: The resulting tensor after gathering and concatenating.
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, ctx.gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Split the gradients and scale them according to the specified mode.

        Args:
            ctx: The context object containing information from the forward pass.
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            torch.Tensor: The gradient of the input with respect to the loss.
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Custom autograd function that splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the specified mode.

    """
    @staticmethod
    def symbolic(graph, input_, process_group, dim, split_sizes):
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.split_sizes = split_sizes

        return _split(input_, process_group, dim, ctx.split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None


def all_to_all(
        input_: torch.Tensor,
        process_group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
        scatter_size: Optional[int] = None,
        gather_size: Optional[int] = None
):
    """
    Performs an all-to-all operation on the input tensor. The input tensor is scattered along the specified scatter
    dimension and then gathered along the specified gather dimension.
    This function supports non-uniform scatter and gather sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int, optional): The index of the dimension that needs to be scattered. Defaults to 2.
        gather_dim (int, optional): The index of the dimension that needs to be gathered. Defaults to 1.
        scatter_size (Optional[int]): The size of the scatter dimension. Default is None.
        gather_size (Optional[int]): The size of the gather dimension. Default is None.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, scatter_size, gather_size)


def split_forward_gather_backward(input_, process_group, dim, split_sizes=None, grad_scale="down"):
    """
    Splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the specified mode.
    This function supports non-uniform split sizes.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to split the tensor.
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "down".

    Returns:
        torch.Tensor: The resulting tensor after splitting and keeping only the corresponding chunk.
    """
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[List[int]] = None,
    grad_scale: str = "up"
) -> torch.Tensor:
    """
    Gathers the input tensor from all processes in the model parallel region and concatenates them along the specified
    dimension. During the backward pass, it splits the gradients and scales them according to the specified mode.
    This function supports non-uniform gather sizes.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to concatenate the gathered tensors.
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

    Returns:
        torch.Tensor: The resulting tensor after gathering and concatenating.
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)


def cal_split_sizes(dim_size, world_size):
    """
    Calculate the split sizes for a given dimension size and number of processes.

    This function divides the dimension size into `world_size` parts, distributing any remainder
    among the first few parts.

    Args:
        dim_size (int): The total size of the dimension to be split.
        world_size (int): The number of processes (or parts) to split the dimension into.

    Returns:
        List[int]: A list of integers representing the size of each part after splitting.
    """
    split_size = dim_size // world_size
    remainder = dim_size % world_size
    size_list = [split_size + (1 if i < remainder else 0) for i in range(world_size)]
    return size_list
