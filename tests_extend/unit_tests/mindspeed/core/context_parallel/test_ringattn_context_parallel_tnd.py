import math
import pytest
import torch
from mindspeed import megatron_adaptor
import torch_npu
import torch.distributed as dist

from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest

from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args

from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ulysses,
                                           get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.context_parallel.utils import sbh_to_tnd, tnd_to_sbh
from mindspeed.utils import set_actual_seq_len



DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_index(actual_seq_len, cp_size, cp_rank, dim=0):
    nseq = len(actual_seq_len)
    points = [0] + actual_seq_len

    def chunk_tensor(i):
        start = points[i]
        end = points[i + 1]

        size = (end - start) // (2 * cp_size)
        part1 = torch.arange(start + cp_rank * size, start + (cp_rank + 1) * size)
        part2 = torch.arange(end - (cp_rank + 1) * size, end - cp_rank * size)

        part = torch.cat((part1, part2))
        return part

    chunks = [chunk_tensor(i) for i in range(nseq)]

    return torch.cat(chunks)


def get_data_on_this_cp_rank(data, batch_actual_seq_len, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    bsz = data.shape[1]
    index_lst = [get_index(actual_seq_len, cp_size, cp_rank).to(data.device) for actual_seq_len in batch_actual_seq_len]

    data_lst = [data[:, i, :].index_select(0, index_lst[i]) for i in range(bsz)]             
    data = torch.stack(data_lst, dim=1) # [s/cp, b, h]
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


def run_ringattn_cp(cp_size, dtype, cp_args):
    from megatron.core import mpu
    causal, send_recv_overlap, use_fused_ring_attention_update, cp_window_size, pse_type = cp_args
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    args.cp_window_size = cp_window_size
    args.context_parallel_algo = 'megatron_cp_algo'
    args.use_fused_ring_attention_update = True
    set_args(args)
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = 4, 32, 1024, 128
    scale = 1.0 / math.sqrt(d)

    batch_actual_seq_len = [[256, 512, 640, 1024], [256, 512, 768, 1024], [768, 1024], [512, 768, 1024]]
    actual_seq_len = []
    for i in range(4):
        actual_seq_len += [n + i * 1024 for n in batch_actual_seq_len[i]]

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    q_tnd, k_tnd, v_tnd = sbh_to_tnd(q, n), sbh_to_tnd(k, n), sbh_to_tnd(v, n)

    pse = None
    if pse_type == 2 or pse_type == 3:
        pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(n)

    if causal:
        attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=q.device))
    else:
        attn_mask = None

    if pse is None:

        out_packed = torch_npu.npu_fusion_attention(
            q_tnd, k_tnd, v_tnd, n, 'TND',
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask,
            scale=scale,
            pre_tockens=s,
            next_tockens=0,
            keep_prob=1.,
            inner_precise=0,
            actual_seq_qlen=actual_seq_len,
            actual_seq_kvlen=actual_seq_len,
            sparse_mode=3 if attn_mask is not None else 0
        )
        out = out_packed[0] # TND
        out = tnd_to_sbh(out, b) # SBH
        out.backward(dout)
    else:
        out = npu_fusion_attention( \
            q, k, v, n, 'SBH', \
            pse=pse, \
            pse_type=pse_type, \
            padding_mask=None, \
            atten_mask=attn_mask, \
            scale=scale, \
            pre_tokens=s, \
            next_tokens=0, \
            keep_prob=1., \
            inner_precise=0, \
            sparse_mode=3 if attn_mask is not None else 0
        )[0]
        out.backward(dout)

    out_ref = get_data_on_this_cp_rank(out.clone().detach(), batch_actual_seq_len, cp_size, rank) # SBH
    k_grad_ref = get_data_on_this_cp_rank(k.grad.clone().detach(), batch_actual_seq_len, cp_size, rank) # SBH
    v_grad_ref = get_data_on_this_cp_rank(v.grad.clone().detach(), batch_actual_seq_len, cp_size, rank) # SBH

    local_actual_seq_len = [x // cp_size for x in actual_seq_len]

    q_ = get_data_on_this_cp_rank(q.clone().detach(), batch_actual_seq_len, cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), batch_actual_seq_len, cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), batch_actual_seq_len, cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), batch_actual_seq_len, cp_size, rank)

    set_actual_seq_len(local_actual_seq_len)
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
    cp_para['pse'] = pse
    cp_para['pse_type'] = pse_type


    cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
    cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
    cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
    cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
    cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()


    out_ = ringattn_context_parallel(q_, k_, v_, n, cp_para,
                                         softmax_scale=scale, attn_mask=None,
                                         actual_seq_qlen=local_actual_seq_len, actual_seq_kvlen=local_actual_seq_len)
    out_.backward(dout_)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out_ref, out_, **tols)
    assert torch.allclose(k_.grad, k_grad_ref, **tols)
    assert torch.allclose(v_.grad, v_grad_ref, **tols)



class TestRingAttnCP(DistributedTest):
    world_size = 8

    @pytest.mark.skip(reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True, True, 2, 1), (True, True, False, 2, 1)])
    def test_ringattn_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_ringattn_cp(self.world_size, torch.bfloat16, cp_args)

