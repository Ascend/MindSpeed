# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _make_chunk_sort_map_kernel(
    # pointers
    split_sizes_ptr,
    sorted_indices_ptr,
    dst_rows_ptr,
    # sizes
    num_splits: tl.constexpr,
    num_tokens: int,
    # metas
    BLOCK_SIZE: int,
    SUB_BLOCK_SIZE: tl.constexpr,
    IDX_LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_start = pid * BLOCK_SIZE

    load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
    load_split_offset_cmp = load_split_offset.to(tl.float32)
    sorted_indices = tl.load(
        sorted_indices_ptr + load_split_offset, mask=load_split_offset_cmp < num_splits
    )
    input_split_sizes = tl.load(
        split_sizes_ptr + load_split_offset, mask=load_split_offset_cmp < num_splits, other=0
    ).to(tl.float32)
    input_split_sizes_cumsum = tl.cumsum(input_split_sizes)
    output_split_sizes = tl.gather(input_split_sizes, sorted_indices, 0).to(tl.float32)

    for i in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        token_offsets = tl.arange(0, SUB_BLOCK_SIZE) + pid_start + i
        token_offsets_cmp = token_offsets.to(tl.float32)

        input_split_sizes_mask = tl.where(input_split_sizes_cumsum[None, :] <= token_offsets_cmp[:, None], 1.0, 0.0)
        input_chunk_indices = tl.sum(input_split_sizes_mask, axis=-1)

        output_chunk_mask = (sorted_indices[None, :] == input_chunk_indices[:, None])
        output_chunk_indices = tl.argmax(output_chunk_mask, axis=-1)
        output_chunk_indices_cmp = output_chunk_indices.to(tl.float32)

        output_pre_split_sizes = tl.where(load_split_offset_cmp[None, :] < output_chunk_indices_cmp[:, None],
            output_split_sizes[None, :], 0.0)

        output_presums = tl.sum(output_pre_split_sizes, axis=-1)
        input_split_sizes_presums = tl.sum(input_split_sizes[None, :] * input_split_sizes_mask, axis=-1)
        dst_rows = output_presums + token_offsets_cmp - input_split_sizes_presums

        store_mask = (token_offsets < num_tokens) & (token_offsets < pid_start + BLOCK_SIZE)
        tl.store(dst_rows_ptr + token_offsets, dst_rows, mask=store_mask)


def make_chunk_sort_map(
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    num_splits: int,
):
    """
    Make a row_id_map for chunk sort.

    Parameters
    ----------
    split_sizes: torch.Tensor
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices: torch.Tensor
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens: int
        Number of tokens in the input tensor.
    num_splits: int
        Number of splits of split_sizes and sorted_indices.
    """
    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device="npu")
    num_blocks = 48
    block_size = triton.cdiv(num_tokens, num_blocks)
    grid = (num_blocks, 1, 1)
    _make_chunk_sort_map_kernel[grid](
        split_sizes,
        sorted_indices,
        row_id_map,
        num_splits,
        num_tokens,
        block_size,
        SUB_BLOCK_SIZE=40,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )
    return row_id_map


@triton.jit
def _sort_chunks_by_map_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    num_tokens: int,
    hidden_size: tl.constexpr,
    block_size: int,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    FORWARD: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_start = pid * block_size
    pid_end = min(pid_start + block_size, num_tokens)

    for i in range(pid_start, pid_end):
        if FORWARD:
            src_row = i
            dst_row = tl.load(row_id_map_ptr + i).to(tl.int64)
        else:
            src_row = tl.load(row_id_map_ptr + i).to(tl.int64)
            dst_row = i

        current_offset = tl.arange(0, hidden_size)
        input_offsets = src_row * stride_input_token + current_offset * stride_input_hidden
        output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden

        inp = tl.load(input_ptr + input_offsets)
        tl.store(output_ptr + output_offsets, inp)

        if PERMUTE_PROBS:
            prob_off = src_row * stride_probs_token
            permuted_prob_off = dst_row * stride_permuted_probs_token
            prob = tl.load(probs_ptr + prob_off)
            tl.store(permuted_probs_ptr + permuted_prob_off, prob)


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
):
    """
    Sort chunks with row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs: torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens: int
        Number of tokens in the input tensor.
    hidden_size: int
        Hidden size of the input tensor.
    is_forward: bool
        Whether the sort is for forward or backward.
    """
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="npu")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="npu")
    else:
        permuted_probs = None

    num_blocks = 48
    block_size = triton.cdiv(num_tokens, num_blocks)
    grid = (num_blocks, 1, 1)
    _sort_chunks_by_map_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        num_tokens,
        hidden_size,
        block_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
        FORWARD=is_forward,
    )

    return output, permuted_probs
 

class _moe_chunk_sort(torch.autograd.Function):
    """functional MoE chunk permute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            return inp, probs

        num_tokens, hidden_size = inp.shape
        num_splits = split_sizes.size(0)
        if num_splits != sorted_idxs.size(0):
            raise AssertionError('sorted_idxs size is not equal to num_splits.')

        if inp.device.type != 'npu':
            raise AssertionError('This operation needs NPU')
        if split_sizes.device.type != 'npu':
            raise AssertionError('This operation needs NPU')
        if sorted_idxs.device.type != 'npu':
            raise AssertionError('This operation needs NPU')
        if probs is not None: 
            if probs.device.type != 'npu':
                raise AssertionError('This operation needs NPU')

        row_id_map = make_chunk_sort_map(
            split_sizes,
            sorted_idxs,
            num_tokens,
            num_splits,
        )
        output, permuted_probs = sort_chunks_by_map(
            inp,
            row_id_map,
            probs,
            num_tokens,
            hidden_size,
            is_forward=True,
        )

        ctx.save_for_backward(row_id_map)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, permuted_probs_grad

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            act_grad, probs_grad = sort_chunks_by_map(
                permuted_act_grad,
                row_id_map,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.hidden_size,
                is_forward=False,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    split_sizes: torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices: torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, _ = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, None)
    return output


def moe_sort_chunks_by_index_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor and probs based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens]. It will be permuted with the tokens according to
        the split_sizes and sorted_indices.
    split_sizes: torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices: torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, permuted_probs = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, probs)
    return output, permuted_probs