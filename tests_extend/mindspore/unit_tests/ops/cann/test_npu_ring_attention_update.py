# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest

import torch
import torch_npu

from mindspeed import megatron_adaptor
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from mindspeed.core.context_parallel.utils import forward_update
from tests_extend.unit_tests.common import TOL_MAPPING

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def create_test_args(use_fused_ring_attention_update=False):
    args = parse_args(None, True)
    args.use_fused_ring_attention_update = use_fused_ring_attention_update
    return args


def run_one_case(batch_size, head_num, seq_size, head_dim, data_type):
    prev_attn_out = torch.rand(seq_size, batch_size, head_num * head_dim, dtype=data_type)
    prev_softmax_max = torch.rand(batch_size, head_num, seq_size, 1, dtype=torch.float32).repeat(1, 1, 1, 8)
    prev_softmax_sum = torch.rand(batch_size, head_num, seq_size, 1, dtype=torch.float32).repeat(1, 1, 1, 8)
    cur_attn_out = torch.rand(seq_size, batch_size, head_num * head_dim, dtype=data_type)
    cur_softmax_max = torch.rand(batch_size, head_num, seq_size, 1, dtype=torch.float32).repeat(1, 1, 1, 8)
    cur_softmax_sum = torch.rand(batch_size, head_num, seq_size, 1, dtype=torch.float32).repeat(1, 1, 1, 8)

    prev_attn_out_fused = prev_attn_out.clone().detach().npu()
    prev_softmax_max_fused = prev_softmax_max.clone().detach().npu()
    prev_softmax_sum_fused = prev_softmax_sum.clone().detach().npu()
    cur_attn_out_fused = cur_attn_out.clone().detach().npu()
    cur_softmax_max_fused = cur_softmax_max.clone().detach().npu()
    cur_softmax_sum_fused = cur_softmax_sum.clone().detach().npu()

    args = create_test_args(False)
    set_args(args)
    attn_out, softmax_max, softmax_sum = forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                                        cur_attn_out, cur_softmax_max, cur_softmax_sum)

    args = create_test_args(True)
    set_args(args)
    attn_out_fused, softmax_max_fused, softmax_sum_fused = forward_update(
        prev_attn_out_fused, prev_softmax_max_fused, prev_softmax_sum_fused,
        cur_attn_out_fused, cur_softmax_max_fused, cur_softmax_sum_fused)

    tols = TOL_MAPPING.get(data_type)
    assert torch.allclose(softmax_max.npu(), softmax_max_fused, **tols)
    assert torch.allclose(softmax_sum.npu(), softmax_sum_fused, **tols)
    assert torch.allclose(attn_out.npu(), attn_out_fused, **tols)


class TestNpuFusedRingAttentionUpdate():

    @pytest.mark.parametrize("bs_hn_seq_hd", [(1, 64, 8192, 128), (1, 32, 8192, 128), (1, 32, 65536, 128), (1, 32, 32768, 128)])
    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float, torch.float16])
    def test_npu_fused_ring_attention_update(self, bs_hn_seq_hd, dtype):
        """
        mixtral： B = 1, N = 32, S = 8192, D = 128
        extend：  B = 1, N = 64, S = 8192, D = 128
        gpt_moe： B = 1, N = 32, S = 65536, D = 128
        llama：   B = 1, N = 32, S = 32768, D = 128
        """
        batch_size, head_num, seq_size, head_dim = bs_hn_seq_hd
        run_one_case(batch_size, head_num, seq_size, head_dim, dtype)
