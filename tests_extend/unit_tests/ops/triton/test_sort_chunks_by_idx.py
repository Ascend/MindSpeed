# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
from mindspeed.lite.ops.triton.sort_chunks_by_idx import make_chunk_sort_map, sort_chunks_by_map


def accuracy_comparison(y_cal, y_ref):
    """
    精度比对函数：根据数据类型选择合适的比对策略。

    不同数据类型的处理策略：
    - 浮点类型（float16/32, bfloat16）：使用 torch.testing.assert_close，设置相对/绝对误差容限
    - 整数类型（int8/16/32/64）：要求完全相等（torch.equal）
    - 布尔类型（bool）：CPU 上严格比较（避免设备差异）
    """
    # 检查输出数据类型是否一致
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # 将张量移动到 NPU（假设测试在 NPU 上进行）
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # 根据数据类型选择不同的比对方式
    if tensor_dtype == torch.float16:
        # float16 精度较低，允许稍大误差
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16 精度更低，建议转为 float32 再比较
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # float32 精度较高，使用更严格的容差
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # 整数类型应完全相等
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # 布尔类型建议在 CPU 上比较，避免设备间布尔表示差异
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


@triton.jit
def _make_chunk_sort_map_kernel_gpu(
    # pointers
    split_sizes_ptr,
    sorted_indices_ptr,
    dst_rows_ptr,
    # sizes
    num_splits: tl.constexpr,
    # metas
    IDX_LOAD_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)

    load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
    sorted_indices = tl.load(
        sorted_indices_ptr + load_split_offset, mask=load_split_offset < num_splits
    )

    # get chunk idx of the current token in the input tensor
    input_split_sizes = tl.load(
        split_sizes_ptr + load_split_offset, mask=load_split_offset < num_splits, other=0
    ).to(tl.int32)
    input_split_sizes_cumsum = tl.cumsum(input_split_sizes)
    input_split_sizes_mask = tl.where(input_split_sizes_cumsum <= pid, 1, 0)
    input_chunk_idx = tl.sum(input_split_sizes_mask)
    input_split_sizes_presum = tl.sum(input_split_sizes * input_split_sizes_mask)
    in_chunk_offset = pid - input_split_sizes_presum

    # get chunk idx of the current token in the output tensor
    output_chunk_mask = tl.where(sorted_indices == input_chunk_idx, 1, 0)
    output_chunk_idx = tl.argmax(output_chunk_mask, axis=-1)

    # make row_id_map
    output_split_sizes = tl.load(
        split_sizes_ptr + sorted_indices, mask=load_split_offset < num_splits
    ).to(tl.int32)
    output_pre_split_sizes = tl.where(load_split_offset < output_chunk_idx, output_split_sizes, 0)
    dst_row = tl.sum(output_pre_split_sizes) + in_chunk_offset
    tl.store(dst_rows_ptr + pid, dst_row)


@triton.jit
def _sort_chunks_by_map_kernel_gpu(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    hidden_size: tl.constexpr,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FORWARD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if FORWARD:
        src_row = pid_t.to(tl.int64)
        dst_row = tl.load(row_id_map_ptr + pid_t).to(tl.int64)
    else:
        src_row = tl.load(row_id_map_ptr + pid_t).to(tl.int64)
        dst_row = pid_t.to(tl.int64)
    current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = current_offset < hidden_size
    input_offsets = src_row * stride_input_token + current_offset * stride_input_hidden
    output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden
    inp = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + output_offsets, inp, mask=mask)
    if PERMUTE_PROBS:
        if pid_h == 0:
            prob_off = src_row * stride_probs_token
            prob = tl.load(probs_ptr + prob_off)
            permuted_prob_off = dst_row * stride_permuted_probs_token
            tl.store(permuted_probs_ptr + permuted_prob_off, prob)


try:
    _sort_chunks_by_map_kernel_gpu = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_sort_chunks_by_map_kernel_gpu)
except RuntimeError:
    pass
    

def make_chunk_sort_map_gpu(
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
    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device='npu')
    grid = (num_tokens,)
    _make_chunk_sort_map_kernel_gpu[grid](
        split_sizes,
        sorted_indices,
        row_id_map,
        num_splits,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )
    return row_id_map


def sort_chunks_by_map_gpu(
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
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device='npu')
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device='npu')
    else:
        permuted_probs = None
    # pylint: disable=unnecessary-lambda-assignment

    def get_grid(META):
        return (num_tokens, triton.cdiv(hidden_size, META["BLOCK_SIZE"]))

    _sort_chunks_by_map_kernel_gpu[get_grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        hidden_size,
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


def gen_split_sizes(num_tokens, num_splits):
    random_numbers = torch.randint(0, num_tokens, (num_splits,), device='npu')
    total_sum = torch.sum(random_numbers)
    scaled_numbers = (random_numbers * num_tokens / total_sum).int()
    scaled_numbers[-1] += num_tokens - torch.sum(scaled_numbers)
    return scaled_numbers

TEST_CASES = [
    (16, 2048, 256),
    (32, 4096, 128)
]


@pytest.mark.parametrize(
    "num_splits,num_tokens,hidden_size",
    [pytest.param(*case, id=f"split{case[0]}-tokens{case[1]}-hid{case[2]}") for case in TEST_CASES]
)
@pytest.mark.skip(reason="Hanged to be fixed")
def test_sort_chunks_by_idx(num_splits, num_tokens, hidden_size):
    split_sizes = gen_split_sizes(num_tokens, num_splits)
    sorted_indices = torch.randperm(num_splits, device='npu')

    ref_row_id_map = make_chunk_sort_map_gpu(split_sizes, sorted_indices, num_tokens, num_splits)
    row_id_map = make_chunk_sort_map(split_sizes, sorted_indices, num_tokens, num_splits)

    accuracy_comparison(ref_row_id_map, row_id_map)

    inp = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device='npu')
    probs = torch.rand(num_tokens, dtype=torch.float32, device='npu')
    
    # test forward
    ref_output, ref_permuted_probs = sort_chunks_by_map_gpu(inp, ref_row_id_map, probs, num_tokens, hidden_size, True)
    output, permuted_probs = sort_chunks_by_map(inp, row_id_map, probs, num_tokens, hidden_size, True)

    accuracy_comparison(ref_output, output)
    accuracy_comparison(ref_permuted_probs, permuted_probs)
    
    # test backward
    ref_output, ref_permuted_probs = sort_chunks_by_map_gpu(inp, ref_row_id_map, probs, num_tokens, hidden_size, False)
    output, permuted_probs = sort_chunks_by_map(inp, row_id_map, probs, num_tokens, hidden_size, False)

    accuracy_comparison(ref_output, output)
    accuracy_comparison(ref_permuted_probs, permuted_probs)
