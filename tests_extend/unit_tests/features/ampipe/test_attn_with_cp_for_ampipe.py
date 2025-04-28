# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import pytest
import torch
import torch_npu
import torch.distributed as dist

from mindspeed import megatron_adaptor
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from unit_tests.common import DistributedTest
from commons import set_random_seed, initialize_model_parallel

from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel_for_ampipe import attn_with_cp_for_ampipe_forward, \
    attn_with_cp_for_ampipe_backward
from mindspeed.moe.ampipe.ampipe_args import FACpFwdArgs, FlashAttentionFwdArgs, FlashAttentionBwdArgs


class AttentionForAmpipe(torch.autograd.Function):
    """Attention implementation with context parallelism"""
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = True
    args.ampipe_degree = 2
    set_args(args)

    @staticmethod
    def forward(ctx, q, k, v, n):

        kv_list = []
        o_max_sum_list = []
        save_tensor_list = []
        ampipe_idx = 0
        fa_cp_fwd_args = FACpFwdArgs(q, k, v)
        fa_fwd_args = FlashAttentionFwdArgs(save_tensor_list, None, n, 0.0, 0, cur_degree=ampipe_idx,
                                            kv_list=kv_list, o_max_sum_list=o_max_sum_list)
        attn_out_a = attn_with_cp_for_ampipe_forward(ctx, fa_cp_fwd_args, fa_fwd_args)

        fa_fwd_args.cur_degree = 1
        attn_out_b = attn_with_cp_for_ampipe_forward(ctx, fa_cp_fwd_args, fa_fwd_args)

        attn_out = torch.cat((attn_out_a, attn_out_b), dim=0)
        attn_out_all = torch.cat((attn_out_a.unsqueeze(0), attn_out_b.unsqueeze(0)), dim=0)
        ctx.save_for_backward(attn_out_all, *save_tensor_list)
        return attn_out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.view(2, dout.shape[0] // 2, *dout.shape[1:])
        attn_out, *saved_tensor_list = ctx.saved_tensors
        ampipe_idx = 0

        fa_bwd_args = FlashAttentionBwdArgs([], None, None, saved_tensor_list, [], cur_degree=ampipe_idx)
        fa_bwd_args.kv_list = []
        fa_bwd_args.dkv_list = []
        fa_bwd_args.dout_list = []

        dq, dk, dv = attn_with_cp_for_ampipe_backward(ctx, attn_out, saved_tensor_list, dout[1], fa_bwd_args)

        fa_bwd_args.cur_degree = 1

        dq, dk, dv = attn_with_cp_for_ampipe_backward(ctx, attn_out, saved_tensor_list, dout[0], fa_bwd_args)

        return dq, dk, dv, None


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim + 1:])
    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=data.device)
    data = data.index_select(dim, index)
    data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    return data


def get_data_on_all_cp_ranks(data, cp_size, dim=0):
    """ Combine data along sequence dimension from multiple chunks.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, -1, *data.shape[dim + 1:])
    index = [[i, 2 * cp_size - i - 1] for i in range(cp_size)]
    index = torch.tensor(index).flatten().to(data.device)
    index = index[:, None, None, None].repeat(1, *data.shape[1:])
    out = torch.empty_like(data)
    out = out.scatter(dim=0, index=index, src=data)
    out = out.view(-1, *out.shape[2:])
    return out


def run_attn_cp_for_ampipe(cp_size, bs, seq_len, dtype):
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=q.device))

    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=3
    )[0]
    out.backward(dout)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    out_ = AttentionForAmpipe.apply(q_, k_, v_, n)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, cp_size)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


class TestRingAttnCP(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
    def test_attn_cp_for_ampipe_seq8192_bs2_fp16(self, dtype):
        run_attn_cp_for_ampipe(self.world_size, 2, 8192, dtype)