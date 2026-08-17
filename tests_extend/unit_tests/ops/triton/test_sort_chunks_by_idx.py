# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from mindspeed.lite.ops.triton.sort_chunks_by_idx import (
    moe_sort_chunks_by_index_with_probs,
)
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs


def gen_split_sizes(num_tokens, num_splits):
    random_numbers = torch.randint(0, num_tokens, (num_splits,), device='npu')
    total_sum = torch.sum(random_numbers)
    scaled_numbers = (random_numbers * num_tokens / total_sum).int()
    scaled_numbers[-1] += num_tokens - torch.sum(scaled_numbers)
    return scaled_numbers


TEST_CASES = [(16, 2048, 256), (32, 4096, 128), (1024, 600000, 7168)]


@pytest.mark.parametrize(
    "num_splits,num_tokens,hidden_size",
    [pytest.param(*case, id=f"split{case[0]}-tokens{case[1]}-hid{case[2]}") for case in TEST_CASES],
)
def test_sort_chunks_by_idx(num_splits, num_tokens, hidden_size):
    split_sizes = gen_split_sizes(num_tokens, num_splits)
    sorted_indices = torch.randperm(num_splits, device='npu')

    inp = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device='npu')
    probs = torch.rand(num_tokens, dtype=torch.float32, device='npu')

    output, permuted_probs = moe_sort_chunks_by_index_with_probs(inp, probs, split_sizes, sorted_indices)

    ref_output, ref_permuted_probs = sort_chunks_by_idxs(inp, split_sizes, sorted_indices, probs, fused=False)

    torch.testing.assert_close(output, ref_output, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(permuted_probs, ref_permuted_probs, rtol=1e-4, atol=1e-4)

    output.backward(torch.ones_like(output))
    npu_grad = inp.grad.clone()

    ref_output.backward(torch.ones_like(ref_output))
    ref_grad = inp.grad.clone()

    torch.testing.assert_close(npu_grad, ref_grad, rtol=1e-4, atol=1e-4)
