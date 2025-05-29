# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest

import torch
import torch_npu

from mindspeed import megatron_adaptor
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.moe_utils import unpermute
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute

from tests_extend.unit_tests.common import TOL_MAPPING


@pytest.mark.skip(reason='this UT need update for new Meagatron version')
class TestNpuFusedPermuteAndUnpermute():

    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_permute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        token_ori = torch.randn(num_tokens, hidden_size).npu().to(dtype).requires_grad_(True)
        indices_ori = torch.randint(0, num_experts, (num_tokens, topk)).npu()

        token_ori = token_ori.requires_grad_(True)
        token_fused = token_ori.clone().detach().requires_grad_(True)
        indices_fused = indices_ori.clone().detach()

        permuted_tokens_ori, sorted_indices_ori = permute(token_ori, indices_ori)
        permuted_tokens_fused, sorted_indices_fused = npu_moe_token_permute(token_fused, indices_fused)
        permuted_tokens_fused.backward(torch.ones(permuted_tokens_fused.shape).to(torch.bfloat16).npu())

        assert torch.allclose(permuted_tokens_ori, permuted_tokens_fused, **tols)
        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True).to(sorted_indices_fused.dtype)
        assert torch.equal(sorted_indices_ori, sorted_indices_fused)

    @pytest.mark.skip(reason='this UT need update for new cann version')
    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        permuted_tokens_ori = torch.randn(num_tokens * topk, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        sorted_indices_ori = torch.argsort(indices.view(-1), stable=True).npu().to(dtype=torch.int32)
        probs_ori = None
        probs_fused = None
        if topk > 1:
            probs_ori = (torch.ones(num_tokens, topk) / topk).npu().to(dtype).requires_grad_(True)
            probs_fused = probs_ori.clone().detach().requires_grad_(True)
        permuted_tokens_fused = permuted_tokens_ori.clone().detach().requires_grad_(True)
        sorted_indices_fused = sorted_indices_ori.clone().detach()

        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True)
        unpermuted_tokens_ori = unpermute(
            permuted_tokens_ori, sorted_indices_ori, probs=probs_ori)

        unpermuted_tokens_fused = npu_moe_token_unpermute(
            permuted_tokens_fused, sorted_indices_fused, probs=probs_fused)

        unpermuted_tokens_fused.backward(torch.ones(unpermuted_tokens_fused.shape).to(torch.bfloat16).npu())
        assert torch.allclose(unpermuted_tokens_ori, unpermuted_tokens_fused, **tols)

    @pytest.mark.skip(reason='this UT need update for new cann version')
    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_ori_permute_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        tokens = torch.randn(num_tokens, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        probs = None
        if topk > 1:
            probs = (torch.ones_like(indices) / topk).npu().to(dtype)

        permuted_tokens, sorted_indices = permute(tokens, indices)
        unpermuted_tokens = unpermute(permuted_tokens, sorted_indices, probs=probs)

        if topk == 1:
            assert torch.equal(unpermuted_tokens, tokens)
        else:
            assert torch.allclose(unpermuted_tokens, tokens, **tols)

    @pytest.mark.skip(reason='this UT need update for new cann version')
    @pytest.mark.parametrize('num_tokens', [1024, 2048, 8192])
    @pytest.mark.parametrize('hidden_size', [6144, 8192, 12288])
    @pytest.mark.parametrize('topk', [1, 4])
    @pytest.mark.parametrize('num_experts', [4, 128])
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_npu_npu_moe_token_permute_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        tokens = torch.randn(num_tokens, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        probs = None
        if topk > 1:
            probs = (torch.ones_like(indices) / topk).npu().to(dtype)

        permuted_tokens, sorted_indices = npu_moe_token_permute(tokens, indices)
        unpermuted_tokens = npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)

        if topk == 1:
            assert torch.equal(unpermuted_tokens, tokens)
        else:
            assert torch.allclose(unpermuted_tokens, tokens, **tols)
