# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from copy import deepcopy
import sys
import pytest
import torch
import torch_npu
sys.argv.append("--use-fused-rotary-pos-emb")
from mindspeed import megatron_adaptor_v2
from tests_extend_v2.unit_tests.common import DistributedTest
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
sys.argv.remove("--use-fused-rotary-pos-emb")

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def create_test_args(use_fused_rotary_pos_emb=False):
    args = parse_args(None, True)
    args.use_fused_rotary_pos_emb = use_fused_rotary_pos_emb
    return args


class TestNpuFusedRotaryEmbedding(DistributedTest):
    world_size = 1

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
    def test_npu_fused_rotary_pos_embedding(self, dtype):
        t_ori = torch.rand(2, 2, 5, 128).npu().to(dtype)
        freqs_ori = torch.rand(1, 2, 1, 128).npu().to(dtype)

        t_fused = deepcopy(t_ori)
        freqs_fused = deepcopy(freqs_ori)

        rotary_interleaved = False

        args = create_test_args(False)
        set_args(args)
        output_patch_ori = _apply_rotary_pos_emb_bshd(t_ori, freqs_ori, rotary_interleaved)

        args = create_test_args(True)
        set_args(args)
        output_patch_fused = _apply_rotary_pos_emb_bshd(t_fused, freqs_fused, rotary_interleaved)

        tol = 0.004 if dtype == torch.bfloat16 else 0.001
        assert torch.allclose(output_patch_ori, output_patch_fused, rtol=tol, atol=tol)
