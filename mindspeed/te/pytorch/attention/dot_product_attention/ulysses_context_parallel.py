# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from abc import ABC
from typing import Optional, List

import torch
import torch_npu
import torch.distributed as dist

from .backend import get_fa_config
from .utils import prepare_sbhd_format, prepare_thd_format


PERMUTE_DIMS1 = {
    4: (1, 2, 3, 0),
    5: (1, 2, 3, 0, 4),
}


PERMUTE_DIMS2 = {
    4: (1, 2, 0, 3),
    5: (1, 2, 0, 3, 4),
}


def AttnFuncWithCPAndQKVOA2A(
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        attention_dropout,
        softmax_scale,
        deterministic,
        cp_group,
        cp_stream,
        ulysses_comm_para
):
    spg = cp_group
    scatter_idx = ulysses_comm_para.get('scatter_idx')
    if qkv_format == 'thd':
        scatter_idx = 1
    gather_idx = ulysses_comm_para.get('gather_idx')
    seq_world_size = torch.distributed.get_world_size(spg)

    # Handle cases where the sequence length of keys/values needs to be adjusted to match queries.
    if seq_world_size > key_layer.shape[scatter_idx] and query_layer.shape[scatter_idx] % key_layer.shape[scatter_idx] == 0:
        key_layer = key_layer.repeat_interleave(query_layer.shape[scatter_idx] // key_layer.shape[scatter_idx], dim=scatter_idx)
        value_layer = value_layer.repeat_interleave(query_layer.shape[scatter_idx] // value_layer.shape[scatter_idx], dim=scatter_idx)

    # Calculate the gather size using the injected gather size calculator
    gather_size = None

    # The gather size usually refers to the size of the output tensor in the `gather_idx` dimension after
    # the all-to-all communication
    # in shape : e.g.,  [s/p:h:]
    q = all_to_all(query_layer, spg, scatter_idx, gather_idx, gather_size)
    k = all_to_all(key_layer, spg, scatter_idx, gather_idx, gather_size)
    v = all_to_all(value_layer, spg, scatter_idx, gather_idx, gather_size)

    # shape_order获取
    fa_config = get_fa_config(attn_mask_type)

    if qkv_format == 'sbhd':
        q, k, v, n_head = prepare_sbhd_format(q, k, v)
        shape_order = 'SBH'
    elif qkv_format == 'thd':
        n_head, cu_seqlens_q, cu_seqlens_kv = prepare_thd_format(q, cu_seqlens_q, cu_seqlens_kv)
        shape_order = 'TND'
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    context_layer = torch_npu.npu_fusion_attention(
        q, k, v, n_head, shape_order,
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=fa_config['pre_tokens'],
        next_tockens=fa_config['next_tokens'],
        keep_prob=1 - attention_dropout,
        inner_precise=0,
        sparse_mode=fa_config['sparse_mode'],
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_kv
    )[0]

    output = all_to_all(context_layer, spg, gather_idx, scatter_idx, query_layer.size(scatter_idx))

    # out e.g., [s/p::h] or [t/p, h, d]
    return output


def all_to_all(
        input_: torch.Tensor,
        process_group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
        gather_size: Optional[int] = None
):
    """
    Performs an all-to-all operation on the input tensor. The input tensor is scattered along the specified scatter
    dimension and then gathered along the specified gather dimension.
    This function supports both aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int, optional): The index of the dimension that needs to be scattered. Defaults to 2.
        gather_dim (int, optional): The index of the dimension that needs to be gathered. Defaults to 1.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, gather_size)


class _AllToAll(torch.autograd.Function):
    """Custom autograd function that performs an all-to-all communication.
    This function supports both aligned and unaligned data.
    """
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, gather_size=None):
        """
        Forward pass: Perform all-to-all communication by scattering the input tensor along the specified scatter
        dimension and then gathering it along the specified gather dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            scatter_dim (int): The index of the dimension that needs to be scattered.
            gather_dim (int): The index of the dimension that needs to be gathered.
            gather_size (int): The size of the gather dimension.

        Returns:
            torch.Tensor: The resulting tensor after performing the all-to-all operation.
        """
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.scatter_size = input_.size(scatter_dim)
        ctx.gather_dim = gather_dim
        ctx.gather_size = gather_size
        output = _all_to_all(
            input_, process_group, scatter_dim, gather_dim, gather_size
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Perform the reverse all-to-all communication

        Args:
            grad_output (torch.Tensor): The gradient of the output with respect to the loss.

        Returns:
            tuple: The gradient of the input with respect to the loss and `None` for other arguments.
        """
        grad_output = _all_to_all(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
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


def _all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. The function supports aligned and unaligned
    data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        group (dist.ProcessGroup): The process group perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.

    Note:
        - The tensor will be split into `world_size` chunks along the `scatter_dim`. Each process will receive one
          chunk. If the total size of the `scatter_dim` is not divisible by `world_size`, the extra elements will be
          distributed to the first few processes, ensuring that no process receives more than one additional element
          compared to the others.
        - The tensor will be gathered along the `gather_dim`, with each process contributing its part to form the
          final output tensor. The gathering process also supports unaligned data, where the remainder elements
          are distributed to the first few processes.
    """
    if not 3 <= input_.dim() <= 4:
        raise ValueError(f"Input tensor must have 3 or 4 dimensions, got {input_.dim()}")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    gather_mod = gather_size % world_size
    scatter_mod = scatter_size % world_size

    if gather_mod != 0 or scatter_mod != 0:
        # In the case of aligned data (both scatter_size and gather_size are divisible by world_size)
        raise ValueError(f"For aligned data, gather_size and scatter_size must be divisible by world_size")
    return _aligned_all_to_all(input_, group, scatter_dim, gather_dim)


def _aligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension.
    Special note: The function only supports aligned data (both scatter_size and gather_size are divisible by
    world_size)
    """
    world_size = dist.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // world_size
    if scatter_dim == 0:
        input_t = input_.reshape([world_size] + inp_shape).contiguous()
    else:
        input_t = input_.reshape([-1, world_size] + inp_shape[scatter_dim:]).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)

    dist.all_to_all_single(output, input_t, group=group)

    output = output.view([world_size] + inp_shape).contiguous()
    output_dim = output.dim()
    if gather_dim == 1:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[gather_dim], *inp_shape[2:])
        output = output.transpose(0, 1).contiguous()
        # the shape of output is (inp_shape[0], world_size, inp_shape[gather_dim], *inp_shape[2:])
    elif gather_dim == 2:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[1], *inp_shape[gather_dim:])
        output = output.permute(*PERMUTE_DIMS2[output_dim]).contiguous()
        # the shape of output is (inp_shape[0], inp_shape[1], world_size, *inp_shape[gather_dim:])
    elif gather_dim == 3:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[gather_dim])
        output = output.permute(*PERMUTE_DIMS1[output_dim]).contiguous()
        # the shape of output is (inp_shape[0], inp_shape[1], inp_shape[2], world_size, inp_shape[gather_dim])
    # The last case: gather_dim == 0:
    # the shape of input_t is (world_size, inp_shape[gather_dim], inp_shape[0], *inp_shape[1:])
    # output requires no action
    # the shape of output is (world_size, inp_shape[gather_dim], inp_shape[0], *inp_shape[1:])
    output = output.view(inp_shape[:gather_dim] + [inp_shape[gather_dim] * world_size, ] + inp_shape[gather_dim + 1:]
                         ).contiguous()

    return output