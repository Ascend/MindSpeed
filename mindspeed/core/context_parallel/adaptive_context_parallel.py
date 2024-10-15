# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu
from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update


def flash_attn_p2p_communicate(scheduling_info, send_q_dst, recv_q_src, send_kv_dst, recv_kv_src, cp_group, rank):
    send_recv_ops = []
    is_recv_q = False
    is_recv_kv = False
    for send_dst in scheduling_info.send_q_dst:
        send_op = torch.distributed.P2POp(torch.distributed.isend, send_q_dst, send_dst, group=cp_group, tag=send_dst)
        send_recv_ops.append(send_op)
    for send_dst in scheduling_info.send_kv_dst:
        send_op = torch.distributed.P2POp(torch.distributed.isend, send_kv_dst, send_dst, group=cp_group, tag=send_dst)
        send_recv_ops.append(send_op)
    if scheduling_info.recv_q_src > -1:
        recv_src = scheduling_info.recv_q_src
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_q_src, recv_src, group=cp_group, tag=rank)
        send_recv_ops.append(recv_op)
        is_recv_q = True
    if scheduling_info.recv_kv_src > -1:
        recv_src = scheduling_info.recv_kv_src
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_kv_src, recv_src, group=cp_group, tag=rank)
        send_recv_ops.append(recv_op)
        is_recv_kv = True
    send_recv_ops_qkv = []
    if len(send_recv_ops) > 0:
        send_recv_ops_qkv = torch.distributed.batch_isend_irecv(send_recv_ops)
    return is_recv_q, is_recv_kv, send_recv_ops_qkv


def flash_attn_p2p_communicate_o(scheduling_info, send_o_dst, recv_o_src, cp_group, rank):
    send_recv_ops = []
    is_recv_o = False
    for recv_src in scheduling_info.recv_o_src:
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_o_src, recv_src, group=cp_group, tag=100000 + rank)
        send_recv_ops.append(recv_op)
        is_recv_o = True
    if scheduling_info.send_o_dst > -1:
        send_dst = scheduling_info.send_o_dst
        send_op = torch.distributed.P2POp(torch.distributed.isend, send_o_dst, send_dst, group=cp_group, tag=100000 + send_dst)
        send_recv_ops.append(send_op)
    send_recv_ops_o = []
    if len(send_recv_ops) > 0:
        send_recv_ops_o = torch.distributed.batch_isend_irecv(send_recv_ops)
    return is_recv_o, send_recv_ops_o


class AdaptiveAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.):
        keep_prob = 1. - dropout_p
        cp_size = cp_para.get("cp_size")
        rank = cp_para.get("rank")
        scheduling_info = cp_para.get('scheduling_info')
        cp_group = cp_para.get('cp_group')

        seq_len = q.shape[0]
        batch_size = q.shape[1]
        head_dim = q.shape[-1] // n

        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)
        send_kv_dst = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, s, b, h]
        recv_q_src, recv_kv_src = None, None
        send_recv_ops_qkv = []
        is_recv_q, is_recv_kv = False, False
        send_o_dst, recv_o_src = None, None
        send_recv_ops_o = []
        is_recv_o, is_send_o = False, False
        attn_out, softmax_max, softmax_sum = None, None, None

        round_num = len(scheduling_info)
        for i in range(round_num + 1):
            is_activate = is_recv_q or is_recv_kv  # receive q or kv last round means calculate this round
            is_send_o = is_recv_q  # receive q last round means send o this round

            # wait until QKV is received
            if len(send_recv_ops_qkv) > 0:
                for send_recv_op in send_recv_ops_qkv:
                    send_recv_op.wait()

            # determine QKV for this round
            cur_q = recv_q_src if is_recv_q else q
            cur_k = recv_kv_src[0] if is_recv_kv else k
            cur_v = recv_kv_src[1] if is_recv_kv else v

            # send QKV for next round
            if i < round_num - 1:
                recv_q_src = torch.empty_like(q)
                recv_kv_src = torch.empty_like(send_kv_dst)
                is_recv_q, is_recv_kv, send_recv_ops_qkv = flash_attn_p2p_communicate(scheduling_info[i],
                                                                                      q, recv_q_src,
                                                                                      send_kv_dst, recv_kv_src,
                                                                                      cp_group, rank)

            # calculate QKV for this round
            if i == 0 or (i < round_num and is_activate):
                this_mask = attn_mask[i] if isinstance(attn_mask, list) else None
                attn_outs = torch_npu.npu_fusion_attention(
                    cur_q, cur_k, cur_v, n, "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=this_mask,
                    scale=softmax_scale,
                    pre_tockens=cur_k.shape[0],
                    next_tockens=cur_k.shape[0],
                    keep_prob=keep_prob,
                    sparse_mode=0
                )
                cur_attn_out, cur_softmax_max, cur_softmax_sum = attn_outs[0], attn_outs[1], attn_outs[2]  # [s, b, h], [b, n, s, 8], [b, n, s, 8]
                if not is_send_o:
                    if i == 0:
                        softmax_sum = cur_softmax_sum
                        softmax_max = cur_softmax_max
                        attn_out = cur_attn_out
                    else:
                        attn_out_updated, softmax_max_updated, softmax_sum_updated = npu_ring_attention_update(
                            attn_out, softmax_max, softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum)
                        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

            # wait until O is received
            if len(send_recv_ops_o) > 0:
                for send_recv_op in send_recv_ops_o:
                    send_recv_op.wait()

            # update O if receive O
            if is_recv_o:
                recv_attn_out = recv_o_src[:, :, :, :head_dim].permute(2, 0, 1, 3)  # [b, n, s, d] -> [s, b, n, d]
                recv_attn_out = recv_attn_out.view(seq_len, batch_size, -1).to(attn_out.dtype)  # [s, b, n, d] -> [s, b, h]
                recv_softmax_max = recv_o_src[:, :, :, head_dim:head_dim + 8]
                recv_softmax_sum = recv_o_src[:, :, :, head_dim + 8:]
                attn_out_updated, softmax_max_updated, softmax_sum_updated = npu_ring_attention_update(
                            attn_out, softmax_max, softmax_sum, recv_attn_out, recv_softmax_max, recv_softmax_sum)
                attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

            # send O for next round
            if i < round_num:
                cur_attn_out = cur_attn_out.view(seq_len, batch_size, n, -1).permute(1, 2, 0, 3) # [s, b, h] -> [s, b, n, d]
                send_o_dst = torch.cat((cur_attn_out, cur_softmax_max), dim=-1) # [s, b, n, d+8]
                send_o_dst = torch.cat((send_o_dst, cur_softmax_sum), dim=-1)  # [s, b, n, d+16]
                recv_o_src = torch.empty_like(send_o_dst)
                is_recv_o, send_recv_ops_o = flash_attn_p2p_communicate_o(scheduling_info[i], send_o_dst, recv_o_src, cp_group, rank)

        k, v = send_kv_dst[0], send_kv_dst[1]
        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]
        ctx.save_for_backward(q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum)
        ctx.n = n
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = rank
        ctx.scheduling_info = scheduling_info
        return attn_out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum = ctx.saved_tensors
        softmax_max = softmax_max.contiguous()
        softmax_sum = softmax_sum.contiguous()

        n = ctx.n
        softmax_scale = ctx.softmax_scale
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        rank = ctx.cp_rank
        dist_attn_scheduler = ctx.scheduling_info

        send_recv_reqs_input = []
        send_recv_reqs_dq = []
        send_recv_reqs_dkv = []
        num_received_dq, num_received_dkv = 0, 0

        # 把m和l的1/8进行all-gather
        softmax_max_all = torch.empty((cp_size, *(softmax_max.shape[:-1])), device=softmax_max.device,
                                      dtype=softmax_max.dtype)
        softmax_sum_all = torch.empty((cp_size, *(softmax_sum.shape[:-1])), device=softmax_sum.device,
                                      dtype=softmax_sum.dtype)
        softmax_max_local = softmax_max[:, :, :, 0].contiguous()  # [b, n, s, 8] -> [b, n, s, 1]
        softmax_sum_local = softmax_sum[:, :, :, 0].contiguous()  # [b, n, s, 8] -> [b, n, s, 1]
        # [b, n, s] -> [8, b, n, s]
        handle_softmax_max = torch.distributed._all_gather_base(softmax_max_all, softmax_max_local,
                                                                group=cp_group, async_op=True)
        handle_softmax_sum = torch.distributed._all_gather_base(softmax_sum_all, softmax_sum_local,
                                                                group=cp_group, async_op=True)

        # 组合需要发送的tensors
        kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, s, b, h]
        qodo = torch.cat((q.unsqueeze(0), attn_out.unsqueeze(0), dout.unsqueeze(0)), dim=0)  # [3, s, b, h]

        # 创建接收tensors的buffer
        kv_recv = torch.empty((2, *kv.shape), device=kv.device, dtype=kv.dtype)  # [2, 2, s, b, h]
        qodo_recv = torch.empty((2, 3, *q.shape), device=q.device, dtype=q.dtype)  # [2, 3, s, b, h]
        dq_recv = torch.empty((2, *q.shape), device=q.device, dtype=q.dtype)  # [2, s, b, h]
        dkv_recv = torch.empty((2, 2, *k.shape), device=k.device, dtype=k.dtype)  # [2, 2, s, b, h]

        # 初始化第0轮的cur_q, cur_k, cur_v, cur_o, cur_do, cur_m, cur_l
        cur_q, cur_k, cur_v = q, k, v
        cur_o, cur_do = attn_out, dout
        cur_m, cur_l = softmax_max, softmax_sum

        dq, dk, dv = None, None, None

        handle_softmax_max.wait()
        handle_softmax_sum.wait()

        # 循环遍历每一个round
        round_cnt = len(dist_attn_scheduler)
        for rnd_idx in range(round_cnt):
            is_active = True
            if len(send_recv_reqs_input) > 0:
                idx = 0
                for send_recv_op in send_recv_reqs_input:
                    send_recv_op.wait()
                    idx += 1

            cur_recv_buf_idx = rnd_idx % 2
            prev_recv_buf_idx = 1 - cur_recv_buf_idx

            # 确定本轮的cur_q, cur_k, cur_v, cur_o, cur_do, cur_m, cur_l
            if rnd_idx > 0:
                prev_scheduling = dist_attn_scheduler[rnd_idx - 1]
                if prev_scheduling.recv_q_src > -1:  # 这一轮计算自己出KV
                    cur_q, cur_o, cur_do = (qodo_recv[prev_recv_buf_idx][0], qodo_recv[prev_recv_buf_idx][1],
                                            qodo_recv[prev_recv_buf_idx][2])
                    cur_k, cur_v = k, v

                    idx = torch.distributed.get_group_rank(cp_group, prev_scheduling.recv_q_src)
                    cur_m = softmax_max_all[idx, :, :, :].view(softmax_max_all.shape[1:] +
                                                                                      (1,)).repeat(1, 1, 1, 8)
                    cur_l = softmax_sum_all[idx, :, :, :].view(softmax_max_all.shape[1:] +
                                                                                      (1,)).repeat(1, 1, 1, 8)
                elif prev_scheduling.recv_kv_src > -1:  # 这一轮计算自己出Q
                    cur_q, cur_o, cur_do = q, attn_out, dout
                    cur_k, cur_v = kv_recv[prev_recv_buf_idx][0], kv_recv[prev_recv_buf_idx][1]
                    cur_m, cur_l = softmax_max, softmax_sum
                else:
                    is_active = False

            # 把本轮的input通信加入input通信队列（需要通信得到下一轮执行所需的q+o+do/k+v、发送下一轮别的device需要的q+o+do/k+v）
            send_recv_ops_input, send_recv_reqs_input = [], []
            cur_scheduling = dist_attn_scheduler[rnd_idx]  # 本轮计算过程中需要并行执行的通信调度

            if cur_scheduling.recv_q_src > -1:
                # recv q + attn_out + dout from cur_scheduling.recv_q_src
                recv_op = torch.distributed.P2POp(torch.distributed.irecv, qodo_recv[cur_recv_buf_idx],
                                                  cur_scheduling.recv_q_src, cp_group, tag=rank)
                send_recv_ops_input.append(recv_op)
            elif cur_scheduling.recv_kv_src > -1:
                # recv kv from cur_scheduling.recv_kv_src
                recv_op = torch.distributed.P2POp(torch.distributed.irecv, kv_recv[cur_recv_buf_idx],
                                                  cur_scheduling.recv_kv_src, cp_group, tag=rank)
                send_recv_ops_input.append(recv_op)

            if len(cur_scheduling.send_q_dst) > 0:
                for send_q_dev in cur_scheduling.send_q_dst:
                    # send q + attn_out + dout to send_q_dev
                    send_op = torch.distributed.P2POp(torch.distributed.isend, qodo, send_q_dev, cp_group,
                                                      tag=send_q_dev)
                    send_recv_ops_input.append(send_op)
            if len(cur_scheduling.send_kv_dst) > 0:
                for send_kv_dev in cur_scheduling.send_kv_dst:
                    # send kv to send_kv_dev
                    send_op = torch.distributed.P2POp(torch.distributed.isend, kv, send_kv_dev, cp_group,
                                                      tag=send_kv_dev)
                    send_recv_ops_input.append(send_op)

            # 发起本轮的input通信
            if len(send_recv_ops_input) > 0:
                send_recv_reqs_input = torch.distributed.batch_isend_irecv(send_recv_ops_input)

            # 仍然按照前向的调度顺序来进行反向的计算，需要q k v do_q m_q l_q
            if is_active:
                this_mask = attn_mask[rnd_idx] if attn_mask is not None else None
                attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                    cur_q, cur_k, cur_v, cur_do, n,
                    "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=this_mask,
                    softmax_max=cur_m,
                    softmax_sum=cur_l,
                    attention_in=cur_o,
                    scale_value=softmax_scale,
                    sparse_mode=0,
                    keep_prob=1.,
                )
                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
            else:
                cur_dq, cur_dk, cur_dv = None, None, None

            if rnd_idx == 0:
                dq = cur_dq
                dk = cur_dk
                dv = cur_dv
            else:
                # 等待output send-recv结束，并用收到的dq/dkdv来更新结果
                if num_received_dq > 0:
                    for send_recv_op in send_recv_reqs_dq:
                        send_recv_op.wait()
                    for i in range(num_received_dq):
                        dq.add_(dq_recv[i])

                if num_received_dkv > 0:
                    for send_recv_op in send_recv_reqs_dkv:
                        send_recv_op.wait()
                    for i in range(num_received_dkv):
                        dk.add_(dkv_recv[i][0])
                        dv.add_(dkv_recv[i][1])
                # 用cur_dq, cur_dk, cur_dv更新结果：检查当前轮的计算是否是帮别人算的，如果是/不是，则加上cur_dk, cur_dv/cur_dq
                send_recv_reqs_dq, send_recv_reqs_dkv = [], []
                send_recv_ops_dq, send_recv_ops_dkv = [], []
                num_received_dq, num_received_dkv = 0, 0
                prev_scheduling = dist_attn_scheduler[rnd_idx - 1]
                if is_active:
                    if prev_scheduling.recv_q_src > -1:  # 这一轮计算自己出KV，是帮别人算
                        dk.add_(cur_dk)
                        dv.add_(cur_dv)
                        send_dq = cur_dq
                        send_op = torch.distributed.P2POp(torch.distributed.isend, send_dq, prev_scheduling.recv_q_src,
                                                          cp_group, tag=rank * 10)
                        send_recv_ops_dq.append(send_op)
                    elif prev_scheduling.recv_kv_src > -1:  # 这一轮计算自己出Q
                        dq.add_(cur_dq)
                        send_dkv = torch.cat((cur_dk.unsqueeze(0), cur_dv.unsqueeze(0)), dim=0)  # [2, s, b, h]
                        send_op = torch.distributed.P2POp(torch.distributed.isend, send_dkv,
                                                          prev_scheduling.recv_kv_src, cp_group, tag=rank * 10)
                        send_recv_ops_dkv.append(send_op)

                # 发起output recv：上一轮发Q的目的device，从那儿收dQ；上一轮发KV的目的device，从那儿收dKdV；
                if len(prev_scheduling.send_q_dst) > 0:
                    for idx, send_q_dst in enumerate(prev_scheduling.send_q_dst):
                        recv_op = torch.distributed.P2POp(torch.distributed.irecv, dq_recv[idx],
                                                          send_q_dst, cp_group, tag=send_q_dst * 10)
                        send_recv_ops_dq.append(recv_op)
                        num_received_dq += 1
                if len(prev_scheduling.send_kv_dst) > 0:
                    for idx, send_kv_dst in enumerate(prev_scheduling.send_kv_dst):
                        recv_op = torch.distributed.P2POp(torch.distributed.irecv, dkv_recv[idx],
                                                          send_kv_dst, cp_group, tag=send_kv_dst * 10)
                        send_recv_ops_dkv.append(recv_op)
                        num_received_dkv += 1

                if len(send_recv_ops_dq) > 0:
                    send_recv_reqs_dq = torch.distributed.batch_isend_irecv(send_recv_ops_dq)
                if len(send_recv_ops_dkv) > 0:
                    send_recv_reqs_dkv = torch.distributed.batch_isend_irecv(send_recv_ops_dkv)

        if num_received_dq > 0:
            for send_recv_op in send_recv_reqs_dq:
                send_recv_op.wait()
            for i in range(num_received_dq):
                dq.add_(dq_recv[i])

        if num_received_dkv > 0:
            for send_recv_op in send_recv_reqs_dkv:
                send_recv_op.wait()
            for i in range(num_received_dkv):
                dk.add_(dkv_recv[i][0])
                dv.add_(dkv_recv[i][1])

        return dq, dk, dv, None, None, None, None, None


def adaptive_attn_context_parallel(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.):
    out = AdaptiveAttention.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p
    )
    return out
