import math
from random import random

import pytest
import torch
import torch_npu
import torch.distributed as dist
from mindspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ulysses,
                                             get_context_parallel_group_for_hybrid_ring,
                                             get_context_parallel_for_hybrid_ring_world_size,
                                             get_context_parallel_for_hybrid_ring_rank,
                                             get_context_parallel_for_hybrid_ring_global_ranks)
from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    data = data.chunk(cp_size, dim=dim)[cp_rank]
    return data


def run_ringattn_cp(cp_size, bs, seq_len, dtype, cp_args):
    from megatron.core import mpu
    causal, send_recv_overlap = cp_args
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    set_args(args)
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    step = seq_len // 4
    random_seq = list(range(step, seq_len + step, step))
    actual_seq_ = [random_seq.copy() for _ in range(b)]
    
    concatenated = []
    for i in range(b):
        this_seq = [p + seq_len * i for p in random_seq] 
        concatenated = concatenated + this_seq


    q_tnd = q.clone().detach().transpose(0, 1).contiguous().reshape((s * b, n, d))
    k_tnd = k.clone().detach().transpose(0, 1).contiguous().reshape((s * b, n, d))
    v_tnd = v.clone().detach().transpose(0, 1).contiguous().reshape((s * b, n, d))

    for t in (q_tnd, k_tnd, v_tnd):
        t.requires_grad = True
    dout_tnd = dout.transpose(0, 1).contiguous().reshape((s * b, n, d))

    std_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=q.device))
    out_tnd = torch_npu.npu_fusion_attention( 
        q_tnd, k_tnd, v_tnd, n, 'TND', 
        pse=None, 
        padding_mask=None, 
        atten_mask=std_mask, 
        scale=scale, 
        pre_tockens=seq_len, 
        next_tockens=0, 
        keep_prob=1., 
        inner_precise=0, 
        actual_seq_qlen=concatenated,
        actual_seq_kvlen=concatenated,
        sparse_mode=3 
    )[0]
    out_tnd.backward(dout_tnd)

    out_ref = out_tnd.reshape((b, s, n * d)).transpose(0, 1).contiguous()

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    in_hybrid_mode = False
    if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
        in_hybrid_mode = True

    if not in_hybrid_mode:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()
    else:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

    cp_para = dict()
    cp_para['causal'] = causal
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None

    out_ = ringattn_context_parallel(q_, k_, v_, n, cp_para, softmax_scale=scale, 
                                    attn_mask=None, actual_seq_qlen=concatenated, actual_seq_kvlen=concatenated)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad_ring = torch.cat(k_grad_list, dim=0)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad_ring = torch.cat(v_grad_list, dim=0)

    kgrad_ref = k_tnd.grad.reshape((b, s, n * d)).transpose(0, 1).contiguous()
    vgrad_ref = v_tnd.grad.reshape((b, s, n * d)).transpose(0, 1).contiguous()
    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out_ref, out_ring, **tols)
    assert torch.allclose(kgrad_ref, k_grad_ring, **tols)
    assert torch.allclose(vgrad_ref, v_grad_ring, **tols)


class TestRingAttnCP(DistributedTest):
    world_size = 8

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(False, False)])


    def test_ringattn_context_parallel_bs2_bf16(self, cp_args):
        run_ringattn_cp(self.world_size, 2, 16384, torch.bfloat16, cp_args)