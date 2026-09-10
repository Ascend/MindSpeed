# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch_npu  # noqa: F401
from mindspeed.lite.ops.triton.sort_chunks_by_idx import (
    make_chunk_sort_map,
    moe_sort_chunks_by_index_with_probs,
)


def sort_chunks_by_idxs_reference(input_tensor, split_sizes, sorted_idxs, probs=None):
    """Reference implementation matching the non-fused small-operator path."""
    input_chunks = torch.split(input_tensor, split_sizes.tolist(), dim=0)
    output = torch.cat([input_chunks[i] for i in sorted_idxs.tolist()], dim=0)

    if probs is not None:
        prob_chunks = torch.split(probs, split_sizes.tolist(), dim=0)
        permuted_probs = torch.cat([prob_chunks[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


def gen_split_sizes(num_tokens, num_splits):
    random_numbers = torch.randint(0, num_tokens, (num_splits,), device='npu')
    total_sum = torch.sum(random_numbers)
    scaled_numbers = (random_numbers * num_tokens / total_sum).int()
    scaled_numbers[-1] += num_tokens - torch.sum(scaled_numbers)
    return scaled_numbers


def _expected_row_id_map(split_sizes, sorted_indices):
    input_starts = []
    input_start = 0
    for size in split_sizes:
        input_starts.append(input_start)
        input_start += size

    result = torch.empty(input_start, dtype=torch.int32)
    output_start = 0
    for expert in sorted_indices:
        size = split_sizes[expert]
        result[input_starts[expert] : input_starts[expert] + size] = torch.arange(
            output_start,
            output_start + size,
            dtype=torch.int32,
        )
        output_start += size
    return result


def _balanced_split_sizes(num_tokens, num_splits):
    base, remainder = divmod(num_tokens, num_splits)
    return [base + int(index < remainder) for index in range(num_splits)]


def _assert_permutation_close(actual, expected):
    """Permutation is copy-only, so any value difference is a real error."""
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


TEST_CASES = [(16, 2048, 256), (32, 4096, 128), (1024, 600000, 7168)]


@pytest.mark.parametrize(
    "num_splits,num_tokens,hidden_size",
    [pytest.param(*case, id=f"split{case[0]}-tokens{case[1]}-hid{case[2]}") for case in TEST_CASES],
)
def test_sort_chunks_by_idx(num_splits, num_tokens, hidden_size):
    split_sizes = gen_split_sizes(num_tokens, num_splits)
    sorted_indices = torch.randperm(num_splits, device='npu')

    inp = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.float32,
        device='npu',
        requires_grad=True,
    )
    probs = torch.rand(num_tokens, dtype=torch.float32, device='npu')
    ref_inp = inp.detach().clone().requires_grad_(True)

    output, permuted_probs = moe_sort_chunks_by_index_with_probs(inp, probs, split_sizes, sorted_indices)

    ref_output, ref_permuted_probs = sort_chunks_by_idxs_reference(
        ref_inp,
        split_sizes,
        sorted_indices,
        probs,
    )

    _assert_permutation_close(output, ref_output)
    _assert_permutation_close(permuted_probs, ref_permuted_probs)

    output.backward(torch.ones_like(output))
    npu_grad = inp.grad.clone()

    ref_output.backward(torch.ones_like(ref_output))
    ref_grad = ref_inp.grad.clone()

    _assert_permutation_close(npu_grad, ref_grad)


@pytest.mark.parametrize(
    "split_sizes,sorted_indices",
    [
        ([0, 1, 3, 0, 13], [4, 2, 0, 3, 1]),
        ([127, 128, 129, 255, 256, 257], [5, 0, 3, 2, 1, 4]),
        (_balanced_split_sizes(32768, 256), list(range(255, -1, -1))),
        (_balanced_split_sizes(43731, 320), list(range(319, -1, -1))),
        (_balanced_split_sizes(65536, 512), list(range(1, 512, 2)) + list(range(0, 512, 2))),
    ],
    ids=[
        "zero-sized-experts",
        "tile-boundaries",
        "tokens32768-experts256",
        "tokens43731-experts320",
        "tokens65536-experts512",
    ],
)
def test_make_chunk_sort_map_expert_tile(split_sizes, sorted_indices):
    split_sizes_npu = torch.tensor(split_sizes, dtype=torch.int32, device="npu")
    sorted_indices_npu = torch.tensor(sorted_indices, dtype=torch.int32, device="npu")
    actual = make_chunk_sort_map(
        split_sizes_npu,
        sorted_indices_npu,
        sum(split_sizes),
        len(split_sizes),
    )
    torch.npu.synchronize()

    expected = _expected_row_id_map(split_sizes, sorted_indices)
    _assert_permutation_close(actual.cpu(), expected)
    _assert_permutation_close(
        torch.sort(actual.cpu()).values,
        torch.arange(sum(split_sizes), dtype=torch.int32),
    )
