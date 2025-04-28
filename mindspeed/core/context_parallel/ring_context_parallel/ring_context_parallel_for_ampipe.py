# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu

from mindspeed.core.parallel_state import get_context_parallel_prev_rank, get_context_parallel_next_rank
from mindspeed.core.context_parallel import get_args
from mindspeed.core.context_parallel import mpu
from mindspeed.core.context_parallel.utils import RingP2P, forward_update


def flash_attention_backward(qkvn, dy, softmax_max, softmax_sum, atten_out, attn_mask=None, softmax_scale=1.,
                             keep_prob=1., seed=0, offset=0, numels=0):
    """FlashAttention backward"""
    q, k, v, n = qkvn
    next_tockens = 0 if attn_mask is not None else k.shape[0]
    sparse_mode = 3 if attn_mask is not None else 0
    output = torch_npu.npu_fusion_attention_grad(
        q, k, v, dy, n,
        "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=attn_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=atten_out,
        scale_value=softmax_scale,
        pre_tockens=k.shape[0],
        next_tockens=next_tockens,
        sparse_mode=sparse_mode,
        keep_prob=keep_prob,
        seed=seed,
        offset=offset,
        numels=numels
    )
    return output


def flash_attention_forward(qkvn, attn_mask=None, softmax_scale=1., keep_prob=1.):
    """FlashAttention forward"""
    q, k, v, n = qkvn
    next_tockens = 0 if attn_mask is not None else k.shape[0]
    sparse_mode = 3 if attn_mask is not None else 0

    output = torch_npu.npu_fusion_attention(
        q, k, v, n, "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=attn_mask,
        scale=softmax_scale,
        pre_tockens=k.shape[0],
        next_tockens=next_tockens,
        keep_prob=keep_prob,
        sparse_mode=sparse_mode
    )
    return output


def attn_with_cp_for_ampipe_forward(ctx, fa_cp_fwd_args,
                                    fa_fwd_args,
                                    dropout_p=0.):
    args = get_args()
    q, k, v = fa_cp_fwd_args.q, fa_cp_fwd_args.k, fa_cp_fwd_args.v
    tensor_list, n = fa_fwd_args.flash_tensor_list, fa_fwd_args.head_num
    kv_list, o_max_sum_list, ampipe_idx = fa_fwd_args.kv_list, fa_fwd_args.o_max_sum_list, fa_fwd_args.cur_degree
    if kv_list is None:
        kv_list = []
    if o_max_sum_list is None:
        o_max_sum_list = []
    keep_prob = 1. - dropout_p
    if args.ampipe_degree > 2:
        raise RuntimeError(f"Context parallel only support ampipe_degree is 2, but got {args.ampipe_degree}")

    head_dim = q.shape[-1] // n
    softmax_scale = head_dim ** (-0.5)

    rank = mpu.get_context_parallel_rank()
    cp_global_ranks = mpu.get_context_parallel_global_ranks()
    prev_rank = get_context_parallel_prev_rank()
    next_rank = get_context_parallel_next_rank()
    cp_size = mpu.get_context_parallel_world_size()
    cp_group = mpu.get_context_parallel_group()
    cp_group_for_send_recv_overlap = mpu.get_context_parallel_group_for_send_recv_overlap() if args.use_cp_send_recv_overlap else cp_group
    send_recv_comm = RingP2P(cp_global_ranks, cp_group, cp_group_for_send_recv_overlap)
    attn_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device)
    attn_mask = torch.triu(attn_mask, diagonal=1)
    if ampipe_idx == 0:
        # split chunk[i]~chunk[2cp-1-i] into chunk[i] and chunk[2cp-1-i],, [2s, b, h] -> [2, s, b, h]
        q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        # (seed, offset, numels) for dropout mask
        rng_states_qa_kva = [[0, 0, 0] for _ in range(cp_size)]
        rng_states_qb_kva = [[0, 0, 0] for _ in range(cp_size)]
        rng_states_qb_kvb = [[0, 0, 0] for _ in range(cp_size)]
        send_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, 2, s, b, h]
        recv_kv = None
        # chunk[i]
        attn_out_a, softmax_max_a, softmax_sum_a = None, None, None
        # chunk[2cp-1-i]
        attn_out_b, softmax_max_b, softmax_sum_b = None, None, None

        for i in range(cp_size):
            # wait until KV is received from recv_src
            if send_recv_comm.wait():
                send_kv = recv_kv
            kv_list.append(send_kv)  # tmp buffer for next ampipe
            if i < cp_size - 1:
                recv_kv = torch.empty_like(send_kv)
                send_recv_comm.async_send_recv(send_kv, recv_kv)
            if i == 0:
                qa, ka, va = [x[0] for x in [q, k, v]]
                qb, kb, vb = [x[1] for x in [q, k, v]]

                attn_outs_a = flash_attention_forward((qa, ka, va, n),
                                                      attn_mask=attn_mask, softmax_scale=softmax_scale,
                                                      keep_prob=keep_prob)
                attn_outs_b = flash_attention_forward((qb, kb, vb, n),
                                                      attn_mask=attn_mask, softmax_scale=softmax_scale,
                                                      keep_prob=keep_prob)
                attn_out_a, softmax_max_a, softmax_sum_a = attn_outs_a[0], attn_outs_a[1], attn_outs_a[2]
                attn_out_b, softmax_max_b, softmax_sum_b = attn_outs_b[0], attn_outs_b[1], attn_outs_b[2]
                # seed, offset, numels (for dropout)
                rng_states_qa_kva[i] = (attn_outs_a[4], attn_outs_a[5], attn_outs_a[6])
                rng_states_qb_kvb[i] = (attn_outs_b[4], attn_outs_b[5], attn_outs_b[6])
            else:
                cur_k, cur_v = send_kv[0], send_kv[1]  # [2, s, b, h]

                if i <= rank:
                    qa, ka, va = [x[0] for x in [q, cur_k, cur_v]]
                    attn_outs_a = flash_attention_forward((qa, ka, va, n),
                                                          attn_mask=None, softmax_scale=softmax_scale,
                                                          keep_prob=keep_prob)
                    cur_attn_out_a, cur_softmax_max_a, cur_softmax_sum_a = attn_outs_a[0], attn_outs_a[1], attn_outs_a[
                        2]
                    rng_states_qa_kva[i] = (attn_outs_a[4], attn_outs_a[5], attn_outs_a[6])
                    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                        attn_out_a, softmax_max_a, softmax_sum_a,
                        cur_attn_out_a, cur_softmax_max_a, cur_softmax_sum_a
                    )
                    attn_out_a, softmax_max_a, softmax_sum_a = attn_out_updated, softmax_max_updated, softmax_sum_updated
                else:
                    kv_idx = i - rank - 1
                    kv = kv_list[kv_idx]
                    cur_k, cur_v = kv[0], kv[1]
                    qb = q[1]
                    ka, va = [x[0] for x in [cur_k, cur_v]]

                    attn_outs_b = flash_attention_forward((qb, ka, va, n),
                                                          attn_mask=None, softmax_scale=softmax_scale)
                    cur_attn_out_b, cur_softmax_max_b, cur_softmax_sum_b = attn_outs_b[0], attn_outs_b[1], attn_outs_b[
                        2]
                    rng_states_qb_kva[kv_idx] = (attn_outs_b[4], attn_outs_b[5], attn_outs_b[6])

                    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                        attn_out_b, softmax_max_b, softmax_sum_b,
                        cur_attn_out_b, cur_softmax_max_b, cur_softmax_sum_b
                    )
                    attn_out_b, softmax_max_b, softmax_sum_b = attn_out_updated, softmax_max_updated, softmax_sum_updated

        attn_out_all = torch.cat((attn_out_a.unsqueeze(0), attn_out_b.unsqueeze(0)), dim=0)
        softmax_max_all = torch.cat((softmax_max_a.unsqueeze(0), softmax_max_b.unsqueeze(0)), dim=0)
        softmax_sum_all = torch.cat((softmax_sum_a.unsqueeze(0), softmax_sum_b.unsqueeze(0)), dim=0)
        o_max_sum_list.append(attn_out_all)
        o_max_sum_list.append(softmax_max_all)
        o_max_sum_list.append(softmax_sum_all)

        k, v = send_kv[0], send_kv[1]
        q, k, v = [x.view(-1, *x.shape[2:]) for x in [q, k, v]]  # [2s, b, h]
        attn_out = attn_out_a
    else:
        q = q.view(2, q.shape[0] // 2, *q.shape[1:])
        qb = q[1]
        attn_out_all, softmax_max_all, softmax_sum_all = o_max_sum_list
        attn_out_b, softmax_max_b, softmax_sum_b = attn_out_all[1], softmax_max_all[1], softmax_sum_all[1]
        rng_states_qa_kva = ctx.rng_states_qa_kva
        rng_states_qb_kva = ctx.rng_states_qb_kva
        rng_states_qb_kvb = ctx.rng_states_qb_kvb

        start_a_idx = cp_size - rank - 1
        start_b_idx = rank + 1

        for i in range(cp_size):
            cur_kv = kv_list[i]
            cur_k, cur_v = cur_kv[0], cur_kv[1]
            if i >= start_a_idx:
                ka, va = cur_k[0], cur_v[0]

                attn_outs_b = flash_attention_forward((qb, ka, va, n),
                                                      attn_mask=None, softmax_scale=softmax_scale)
                cur_attn_out_b, cur_softmax_max_b, cur_softmax_sum_b = attn_outs_b[0], attn_outs_b[1], attn_outs_b[2]
                rng_states_qb_kva[i] = (attn_outs_b[4], attn_outs_b[5], attn_outs_b[6])
                attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                    attn_out_b, softmax_max_b, softmax_sum_b,
                    cur_attn_out_b, cur_softmax_max_b, cur_softmax_sum_b
                )
                attn_out_b, softmax_max_b, softmax_sum_b = attn_out_updated, softmax_max_updated, softmax_sum_updated
            if i >= start_b_idx:
                kb, vb = cur_k[1], cur_v[1]
                attn_outs_b = flash_attention_forward((qb, kb, vb, n),
                                                      attn_mask=None, softmax_scale=softmax_scale)
                cur_attn_out_b, cur_softmax_max_b, cur_softmax_sum_b = attn_outs_b[0], attn_outs_b[1], attn_outs_b[2]
                rng_states_qb_kvb[i] = (attn_outs_b[4], attn_outs_b[5], attn_outs_b[6])
                attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                    attn_out_b, softmax_max_b, softmax_sum_b,
                    cur_attn_out_b, cur_softmax_max_b, cur_softmax_sum_b
                )
                attn_out_b, softmax_max_b, softmax_sum_b = attn_out_updated, softmax_max_updated, softmax_sum_updated
        kv = kv_list[-1]
        k, v = kv[0], kv[1]
        q, k, v = [x.view(-1, *x.shape[2:]) for x in [q, k, v]]  # [2s, b, h]
        attn_out = attn_out_b
        attn_out_all[1], softmax_max_all[1], softmax_sum_all[1] = attn_out_b, softmax_max_b, softmax_sum_b

        tensor_list.extend([q, k, v, attn_mask, softmax_max_all, softmax_sum_all])

        ctx.n = n
        ctx.rank = rank
        ctx.keep_prob = keep_prob
        ctx.cp_size = cp_size
        ctx.cp_group = cp_group
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank
        ctx.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap
        ctx.softmax_scale = softmax_scale
    ctx.rng_states_qa_kva = rng_states_qa_kva
    ctx.rng_states_qb_kva = rng_states_qb_kva
    ctx.rng_states_qb_kvb = rng_states_qb_kvb
    return attn_out


def attn_with_cp_for_ampipe_backward(ctx, attn_out, saved_tensor_list, dout, fa_bwd_args):
    args = get_args()
    kv_list, dkv_list, dout_list, ampipe_idx = (fa_bwd_args.kv_list, fa_bwd_args.dkv_list,
                                                fa_bwd_args.dout_list, fa_bwd_args.cur_degree)

    if kv_list is None:
        kv_list = []
    if dkv_list is None:
        dkv_list = []
    if dout_list is None:
        dout_list = []
    if args.ampipe_degree > 2:
        raise RuntimeError(f"Context parallel only support ampipe_degree is 2, but got {args.ampipe_degree}")

    q, k, v, attn_mask, softmax_max, softmax_sum = saved_tensor_list
    n = ctx.n
    rank = ctx.rank
    softmax_scale = ctx.softmax_scale
    cp_size = ctx.cp_size
    cp_group = ctx.cp_group
    cp_group_for_send_recv_overlap = ctx.cp_group_for_send_recv_overlap
    cp_global_ranks = mpu.get_context_parallel_global_ranks()
    keep_prob = ctx.keep_prob
    rng_states_qa_kva = ctx.rng_states_qa_kva
    rng_states_qb_kva = ctx.rng_states_qb_kva
    rng_states_qb_kvb = ctx.rng_states_qb_kvb
    # [2s, b, h] -> [2, s, b, h]
    q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

    attn_out_a, softmax_max_a, softmax_sum_a = attn_out[0], softmax_max[0], softmax_sum[0]
    attn_out_b, softmax_max_b, softmax_sum_b = attn_out[1], softmax_max[1], softmax_sum[1]

    if ampipe_idx == 0:
        send_recv_comm = RingP2P(cp_global_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)
        dq, dk, dv = None, None, None
        recv_kv_dkv = None
        recv_kv = None
        recv_dkv = None
        # [s, b, h]
        qa, ka, va = [x[0] for x in [q, k, v]]
        qb, kb, vb = [x[1] for x in [q, k, v]]
        dq_b = torch.zeros_like(qb)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_kv_dkv = torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)

        for i in range(cp_size):
            # wait until KV is received from recv_src
            if send_recv_comm.wait():
                # only received kv in the second loop
                if i == 1:
                    send_kv = recv_kv
                    send_kv_dkv[0].copy_(send_kv)
                else:
                    send_kv_dkv = recv_kv_dkv
            if i > 0:
                dkv = torch.cat((dk.unsqueeze(0), dv.unsqueeze(0)), dim=0)
                send_kv_dkv[1].copy_(dkv)

            # just send-recv kv in the first loop
            if i == 0:
                send_kv = kv
                recv_kv = torch.empty_like(send_kv)
                send_recv_comm.async_send_recv(send_kv, recv_kv)
                kv_list.append(send_kv)
            # just send-recv dkv in the last loop
            elif i == cp_size - 1:
                send_dkv = send_kv_dkv[1]
                recv_dkv = torch.empty_like(send_dkv)
                send_recv_comm.async_send_recv(send_dkv, recv_dkv)
                cur_k, cur_v = send_kv_dkv[0][0], send_kv_dkv[0][1]
                ka, va = cur_k[0], cur_v[0]
                kv_list.append(send_kv_dkv[0])
            else:
                recv_kv_dkv = torch.empty_like(send_kv_dkv)
                send_recv_comm.async_send_recv(send_kv_dkv, recv_kv_dkv)
                cur_k, cur_v = send_kv_dkv[0][0], send_kv_dkv[0][1]
                ka, va = cur_k[0], cur_v[0]
                kv_list.append(send_kv_dkv[0])

            attn_grad_outs_b = flash_attention_backward(
                (qb, ka, va, n),
                dout, softmax_max_b, softmax_sum_b, attn_out_b,
                None, softmax_scale, keep_prob, rng_states_qb_kva[cp_size - i - 1][0],
                rng_states_qb_kva[cp_size - i - 1][1], rng_states_qb_kva[cp_size - i - 1][2]
            )

            cur_dq_b, cur_dk_a, cur_dv_a = attn_grad_outs_b[0], attn_grad_outs_b[1], attn_grad_outs_b[2]
            if i == 0:
                dq_b = cur_dq_b
                dk[0].copy_(cur_dk_a)
                dv[0].copy_(cur_dv_a)
            else:
                # wait until dKV is received from recv_src
                send_recv_comm.wait()
                # only received dkv in the last loop
                if i == cp_size - 1:
                    dkv = recv_dkv
                else:
                    send_kv_dkv = recv_kv_dkv
                    dkv = send_kv_dkv[1]
                dk, dv = dkv[0], dkv[1]
                dq_b.add_(cur_dq_b)
                dk[0].add_(cur_dk_a)
                dv[0].add_(cur_dv_a)
        dkv_list.append(dq_b)
        dkv_list.append(dk[0])
        dkv_list.append(dv[0])
        dout_list.append(dout)
    else:
        send_recv_comm = RingP2P(cp_global_ranks, cp_group, cp_group_for_send_recv_overlap)
        kv_list.reverse()

        recv_dkv = None
        # [s, b, h]
        qa, ka, va = [x[0] for x in [q, k, v]]
        qb, kb, vb = [x[1] for x in [q, k, v]]
        dq_a, dk_a, dv_a, dq_b, dk_b, dv_b = [torch.zeros_like(x) for x in [qa, ka, va, qb, kb, vb]]
        send_dkv = torch.empty((2, 2, *ka.shape), dtype=ka.dtype, device=ka.device)

        for i in range(cp_size):
            # the first loop no send-recv
            if i > 0:
                if i <= rank + 1:
                    if i <= rank:
                        dkv_a = torch.cat((dk_a.unsqueeze(0), dv_a.unsqueeze(0)), dim=0)
                        # send_dkv = dkv_a
                        send_dkv[0].copy_(dkv_a)
                    else:
                        dkv_b = torch.cat((dk_b.unsqueeze(0), dv_b.unsqueeze(0)), dim=0)
                        # send_dkv = dkv_b
                        send_dkv[1].copy_(dkv_b)
                else:
                    dkv_a = torch.cat((dk_a.unsqueeze(0), dv_a.unsqueeze(0)), dim=0)
                    dkv_b = torch.cat((dk_b.unsqueeze(0), dv_b.unsqueeze(0)), dim=0)
                    dkv = torch.cat((dkv_a.unsqueeze(0), dkv_b.unsqueeze(0)), dim=0)
                    send_dkv = dkv

                recv_dkv = torch.empty_like(send_dkv)
                send_recv_comm.async_send_recv(send_dkv, recv_dkv)

            if i == cp_size - 1:
                cur_kv = kv_list[0]
                ka, va = cur_kv[0][0], cur_kv[1][0]
                kb, vb = cur_kv[0][1], cur_kv[1][1]
                attn_grad_outs_a = flash_attention_backward(
                    (qa, ka, va, n),
                    dout, softmax_max_a, softmax_sum_a, attn_out_a,
                    attn_mask, softmax_scale, keep_prob,
                    rng_states_qa_kva[0][0], rng_states_qa_kva[0][1], rng_states_qa_kva[0][2]
                )
                attn_grad_outs_b = flash_attention_backward(
                    (qb, kb, vb, n),
                    dout_list[0], softmax_max_b, softmax_sum_b, attn_out_b,
                    attn_mask, softmax_scale, keep_prob,
                    rng_states_qb_kvb[0][0], rng_states_qb_kvb[0][1], rng_states_qb_kvb[0][2]
                )
                cur_dq_a, cur_dk_a, cur_dv_a = attn_grad_outs_a[0], attn_grad_outs_a[1], attn_grad_outs_a[2]
                cur_dq_b, cur_dk_b, cur_dv_b = attn_grad_outs_b[0], attn_grad_outs_b[1], attn_grad_outs_b[2]
            elif i < rank:
                cur_kv = kv_list[i + 1]
                ka, va = cur_kv[0][0], cur_kv[1][0]
                attn_grad_outs_a = flash_attention_backward(
                    (qa, ka, va, n),
                    dout, softmax_max_a, softmax_sum_a, attn_out_a,
                    None, softmax_scale, keep_prob,
                    rng_states_qa_kva[i + 1][0], rng_states_qa_kva[i + 1][1], rng_states_qa_kva[i + 1][2]
                )
                cur_dq_a, cur_dk_a, cur_dv_a = attn_grad_outs_a[0], attn_grad_outs_a[1], attn_grad_outs_a[2]
            else:
                cur_kv = kv_list[i + 1]
                kb, vb = cur_kv[0][1], cur_kv[1][1]
                attn_grad_outs_b = flash_attention_backward(
                    (qb, kb, vb, n),
                    dout_list[0], softmax_max_b, softmax_sum_b, attn_out_b,
                    None, softmax_scale, keep_prob,
                    rng_states_qb_kvb[i + 1][0], rng_states_qb_kvb[i + 1][1], rng_states_qb_kvb[i + 1][2]
                )
                cur_dq_b, cur_dk_b, cur_dv_b = attn_grad_outs_b[0], attn_grad_outs_b[1], attn_grad_outs_b[2]

            if i == 0:
                if rank == 0:
                    dq_b, dk_b, dv_b = cur_dq_b, cur_dk_b, cur_dv_b
                else:
                    dq_a, dk_a, dv_a = cur_dq_a, cur_dk_a, cur_dv_a
            else:
                # wait until dKV is received from recv_src
                send_recv_comm.wait()

                if i < cp_size - 1:
                    if rank == 0:
                        dkv_a = recv_dkv[0]
                        dk_a, dv_a = dkv_a[0], dkv_a[1]

                        dq_b.add_(cur_dq_b)
                        dk_b, dv_b = cur_dk_b, cur_dv_b
                    elif i <= rank:
                        if i == rank:
                            dkv_b = recv_dkv[1]
                            dk_b, dv_b = dkv_b[0], dkv_b[1]

                            dq_b.add_(cur_dq_b)
                            dk_b.add_(cur_dk_b)
                            dv_b.add_(cur_dv_b)
                        else:
                            dkv_a = recv_dkv[0]
                            dk_a, dv_a = dkv_a[0], dkv_a[1]

                            dq_a.add_(cur_dq_a)
                            dk_a.add_(cur_dk_a)
                            dv_a.add_(cur_dv_a)
                    else:
                        dkv = recv_dkv
                        dkv_a, dkv_b = dkv[0], dkv[1]
                        dk_a, dv_a = dkv_a[0], dkv_a[1]
                        dk_b, dv_b = dkv_b[0], dkv_b[1]

                        dq_b.add_(cur_dq_b)
                        dk_b.add_(cur_dk_b)
                        dv_b.add_(cur_dv_b)
                else:
                    prev_dq_b, prev_dk_a, prev_dv_a = dkv_list
                    if rank == 0:
                        dkv_a = recv_dkv[0]
                        dk_a, dv_a = dkv_a[0], dkv_a[1]

                        dq_a = cur_dq_a
                        dk_a.add_(cur_dk_a)
                        dv_a.add_(cur_dv_a)
                        dk_b, dv_b = cur_dk_b, cur_dv_b
                    elif rank == cp_size - 1:
                        dkv_b = recv_dkv[1]
                        dk_b, dv_b = dkv_b[0], dkv_b[1]

                        dq_a.add_(cur_dq_a)
                        dk_a, dv_a = cur_dk_a, cur_dv_a
                        dk_b.add_(cur_dk_b)
                        dv_b.add_(cur_dv_b)
                    else:
                        dkv = recv_dkv
                        dkv_a, dkv_b = dkv[0], dkv[1]
                        dk_a, dv_a = dkv_a[0], dkv_a[1]
                        dk_b, dv_b = dkv_b[0], dkv_b[1]

                        dq_a.add_(cur_dq_a)
                        dk_a.add_(cur_dk_a)
                        dv_a.add_(cur_dv_a)
                        dk_b.add_(cur_dk_b)
                        dv_b.add_(cur_dv_b)

                    dk_a.add_(prev_dk_a)
                    dv_a.add_(prev_dv_a)
                    dq_b.add_(cur_dq_b)
                    dq_b.add_(prev_dq_b)

        dq = torch.cat((dq_a.unsqueeze(0), dq_b.unsqueeze(0)), dim=0)
        dk = torch.cat((dk_a.unsqueeze(0), dk_b.unsqueeze(0)), dim=0)
        dv = torch.cat((dv_a.unsqueeze(0), dv_b.unsqueeze(0)), dim=0)
        dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]

    return dq, dk, dv
