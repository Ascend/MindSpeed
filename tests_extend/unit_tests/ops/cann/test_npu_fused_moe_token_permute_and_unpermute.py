# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest

import torch
import torch_npu

from mindspeed import megatron_adaptor
from unit_tests.common import DistributedTest
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.moe_utils import unpermute


def create_test_args(use_fused_moe_token_permute_and_unpermute=False):
    args = parse_args(None, True)
    args.use_fused_moe_token_permute_and_unpermute = use_fused_moe_token_permute_and_unpermute
    return args


class TestNpuFusedPermuteAndUnpermute(DistributedTest):
    world_size = 1

    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_npu_fused_permute(self, dtype):
        token_ori = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).npu().to(dtype)
        indices_ori = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
        topk = 2
        token_ori = token_ori.requires_grad_(True)
        token_fused = token_ori.clone().detach().requires_grad_(True)
        indices_fused = indices_ori.clone().detach()

        args = create_test_args(False)
        set_args(args)
        permuted_tokens_ori, sorted_indices_ori = permute(token_ori, indices_ori, topk=topk)

        args = create_test_args(True)
        set_args(args)
        permuted_tokens_fused, sorted_indices_fused = permute(token_fused, indices_fused)
        permuted_tokens_fused.backward(torch.ones(permuted_tokens_fused.shape).to(torch.bfloat16).npu())

        tol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(permuted_tokens_ori, permuted_tokens_fused, rtol=tol, atol=tol)
        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True).to(sorted_indices_fused.dtype)
        assert torch.equal(sorted_indices_ori, sorted_indices_fused)

    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_npu_fused_unpermute(self, dtype):
        permuted_tokens_ori = torch.tensor([[1., 1., 1.],
                                            [0., 0., 0.],
                                            [0., 0., 0.],
                                            [3., 3., 3.],
                                            [2., 2., 2.],
                                            [1., 1., 1.],
                                            [2., 2., 2.],
                                            [3., 3., 3.]]).npu().to(dtype)
        sorted_indices_ori = torch.tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=torch.int32).npu()
        indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
        topk = 2
        probs_ori = torch.ones_like(indices) / 2
        probs_ori = probs_ori.npu().to(dtype)
        permuted_tokens_ori = permuted_tokens_ori.requires_grad_(True)
        permuted_tokens_fused = permuted_tokens_ori.clone().detach().requires_grad_(True)
        sorted_indices_fused = sorted_indices_ori.clone().detach()
        probs_fused = probs_ori.clone().detach().requires_grad_(True)

        args = create_test_args(False)
        set_args(args)
        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True)
        unpermuted_tokens_ori = unpermute(
            permuted_tokens_ori, sorted_indices_ori, probs=probs_ori, topk=topk)

        args = create_test_args(True)
        set_args(args)
        unpermuted_tokens_fused = unpermute(
            permuted_tokens_fused, sorted_indices_fused, probs=probs_fused, topk=topk)

        tol = 0.004 if dtype == torch.bfloat16 else 0.001
        unpermuted_tokens_fused.backward(torch.ones(unpermuted_tokens_fused.shape).to(torch.bfloat16).npu())
        assert torch.allclose(unpermuted_tokens_ori, unpermuted_tokens_fused, rtol=tol, atol=tol)

    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_fused_permute_and_unpermute(self, dtype):
        tokens = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).npu().to(dtype)
        indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
        probs = torch.ones_like(indices) / 2
        probs = probs.npu().to(dtype)
        args = create_test_args(True)
        set_args(args)
        permuted_tokens, sorted_indices = permute(tokens, indices)
        unpermuted_tokens_fused = unpermute(permuted_tokens, sorted_indices, probs=probs)
        assert torch.allclose(tokens, unpermuted_tokens_fused)

    @pytest.mark.skip(reason='not support for current version')
    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_ori_permute_and_unpermute(self, dtype):
        tokens = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).npu().to(dtype)
        indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
        topk = 2
        probs = torch.ones_like(indices) / 2
        probs = probs.npu().to(dtype)
        args = create_test_args(False)
        set_args(args)
        permuted_tokens_ori, sorted_indices_ori = permute(tokens, indices, topk=topk)
        unpermuted_tokens_ori = unpermute(permuted_tokens_ori, sorted_indices_ori, probs=probs)

        assert torch.allclose(tokens, unpermuted_tokens_ori)