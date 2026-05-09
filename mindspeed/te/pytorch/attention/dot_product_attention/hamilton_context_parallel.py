# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import List
import torch
import torch_npu
import torch.distributed as dist
from .utils import get_distributed_rank, get_distributed_world_size, general_output_update_for_ha_of_bsh_format, general_output_update_for_ha_of_tnd_format


class AttnFuncWithCPAndKVA2AForSBHD(torch.autograd.Function):
    """
    Attention implementation with context parallelism, which is by transmitting KV among CP ranks using AlltoAll.
    This implementation is for SBHD format
    """
    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            n_head,
            attention_mask,
            qkv_format,
            attn_mask_type,
            attention_dropout,
            softmax_scale,
            deterministic,
            cp_group,
            ha_params
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        if attention_mask is not None or 'causal' in attn_mask_type:
            raise AssertionError("Currently in the implementation of Hamilton Attention, only full attention pattern"
                                 " is supported, please set 'attention_mask' as 'None', and 'attn_mask_type' as 'full'")

        path_num = ha_params.get("path_num", None)
        if not isinstance(path_num, int):
            raise TypeError(f"'path_num' must be type of 'int', but got {type(path_num).__name__}")
        out_mapping = ha_params.get("out_mapping", None)
        if not isinstance(out_mapping, list):
            raise TypeError(f"'out_mapping' must be type of 'list', but got {type(out_mapping).__name__}")
        in_mapping = ha_params.get("in_mapping", None)
        if not isinstance(in_mapping, list):
            raise TypeError(f"'in_mapping' must be type of 'list, but got {type(in_mapping).__name__}'")

        out_mapping_of_this_rank = out_mapping[rank]
        in_mapping_of_this_rank = in_mapping[rank]

        # q, k and v should be of shape: [S, B, H]
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise AssertionError(f"The q/k/v tensor should be of 3 dimensions, but got q.ndim={q.ndim}, "
                                 f"k.ndim={k.ndim}, v.ndim={v.ndim}")

        seq_dim = qkv_format.index("s")
        if not (k.shape[seq_dim] % path_num == 0 and v.shape[seq_dim] % path_num == 0):
            raise AssertionError(f"In hamilton attention, sequence length per device (k's seq_len={q.shape[seq_dim]}, "
                                 f"v's seq_len={v.shape[seq_dim]}) "
                                 f"needs to be divisible by path_num (path_num)!")

        a2a_comm = A2AComm(cp_group)
        k_split = k.chunk(path_num, dim=seq_dim)
        v_split = v.chunk(path_num, dim=seq_dim)
        cur_kv_split = [torch.stack((k_split[i], v_split[i]), dim=0) for i in range(path_num)]
        next_kv_split = [torch.empty_like(cur_kv_split[0]) for _ in range(path_num)]
        cur_kv = None

        global_attn_outs = (None, None, None)
        for j in range(cp_size):
            if j < cp_size - 1:
                kv_send = [cur_kv_split[out_mapping_of_this_rank[i]] if out_mapping_of_this_rank[i] != -1
                    else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                kv_recv = [next_kv_split[in_mapping_of_this_rank[i]] if in_mapping_of_this_rank[i] != -1
                    else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                a2a_comm.all_to_all(kv_send, kv_recv)

            cur_kv = torch.cat(cur_kv_split, dim=seq_dim + 1)
            cur_k, cur_v = cur_kv[0], cur_kv[1]
            attn_outs = torch_npu.npu_fusion_attention(
                q,
                cur_k,
                cur_v,
                n_head,
                'SBH',
                pse=None,
                padding_mask=None,
                atten_mask=None,
                scale=softmax_scale,
                keep_prob=1 - attention_dropout,
                sparse_mode=0
            )
            global_attn_outs = general_output_update_for_ha_of_bsh_format(j, attn_outs, global_attn_outs)

            if a2a_comm.wait():
                cur_kv_split, next_kv_split = next_kv_split, cur_kv_split

        g_attn_out, g_softmax_max, g_softmax_sum = global_attn_outs
        ctx.save_for_backward(q, cur_kv[0].clone(), cur_kv[1].clone(),
                              g_attn_out, g_softmax_max, g_softmax_sum)
        ctx.n_head = n_head
        ctx.attention_dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.ha_params = ha_params
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = rank
        ctx.qkv_format = qkv_format

        return g_attn_out

    @staticmethod
    def backward(ctx, dout):
        qkv_format = ctx.qkv_format
        rank = ctx.cp_rank
        cp_size = ctx.cp_size
        cp_group = ctx.cp_group
        ha_params = ctx.ha_params
        softmax_scale = ctx.softmax_scale
        attention_dropout = ctx.attention_dropout
        n_head = ctx.n_head
        q, k, v, g_attn_out, g_softmax_max, g_softmax_sum = ctx.saved_tensors

        path_num = ha_params.get("path_num", None)
        out_mapping = ha_params.get("out_mapping", None)
        in_mapping = ha_params.get("in_mapping", None)
        out_mapping_of_this_rank = in_mapping[rank]
        in_mapping_of_this_rank = out_mapping[rank]

        kv_a2a_comm = A2AComm(cp_group)
        dkv_a2a_comm = A2AComm(cp_group)

        seq_dim = qkv_format.index("s")
        # k_split's shape: [s/path_num, b, h]
        k_split = k.chunk(path_num, dim=seq_dim)
        v_split = v.chunk(path_num, dim=seq_dim)
        # each element shape: [2, s/path_num, b, h]
        cur_kv_split = [torch.stack((k_split[i], v_split[i]), dim=0) for i in range(path_num)]
        next_kv_split = [torch.empty_like(cur_kv_split[0]) for _ in range(path_num)]
        cur_dkv_split = [torch.zeros_like(cur_kv_split[0]) for _ in range(path_num)]
        next_dkv_split = [torch.zeros_like(cur_kv_split[0]) for _ in range(path_num)]
        cur_dkv = None

        dq = torch.zeros_like(q)
        for j in range(cp_size):
            if j < cp_size - 1:
                kv_send = [cur_kv_split[out_mapping_of_this_rank[i]] if out_mapping_of_this_rank[i] != -1
                           else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                kv_recv = [next_kv_split[in_mapping_of_this_rank[i]] if in_mapping_of_this_rank[i] != -1
                           else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                kv_a2a_comm.all_to_all(kv_send, kv_recv)

            cur_kv = torch.cat(cur_kv_split, dim=seq_dim + 1)
            cur_k, cur_v = cur_kv[0], cur_kv[1]

            attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                q,
                cur_k,
                cur_v,
                dout,
                n_head,
                "SBH",
                pse=None,
                padding_mask=None,
                atten_mask=None,
                softmax_max=g_softmax_max,
                softmax_sum=g_softmax_sum,
                attention_in=g_attn_out,
                scale_value=softmax_scale,
                keep_prob=1 - attention_dropout,
                sparse_mode=0
                )
            dq_step, dk_step, dv_step = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

            # receive dkv
            if j > 0 and dkv_a2a_comm.wait():
                cur_dkv_split, next_dkv_split = next_dkv_split, cur_dkv_split
            # update dq, dk, dv
            cur_dkv = torch.cat(cur_dkv_split, dim=seq_dim + 1)
            dk, dv = cur_dkv[0], cur_dkv[1]
            dq.add_(dq_step)
            dk.add_(dk_step)
            dv.add_(dv_step)
            # prepare for send dk/dv
            cur_dkv_split = cur_dkv.chunk(path_num, dim=seq_dim + 1)
            cur_dkv_split = [x.contiguous() for x in cur_dkv_split]
            # send dk/dv
            if j < cp_size - 1:
                dkv_send = [cur_dkv_split[out_mapping_of_this_rank[i]] if out_mapping_of_this_rank[i] != -1
                            else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                dkv_recv = [next_dkv_split[in_mapping_of_this_rank[i]] if in_mapping_of_this_rank[i] != -1
                            else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                dkv_a2a_comm.all_to_all(dkv_send, dkv_recv)

            # receive kv
            if kv_a2a_comm.wait():
                cur_kv_split, next_kv_split = next_kv_split, cur_kv_split

        # receive the last dk/dv
        if dkv_a2a_comm.wait():
            cur_dkv_split, next_dkv_split = next_dkv_split, cur_dkv_split
            cur_dkv = torch.cat(cur_dkv_split, dim=seq_dim + 1)
        dk, dv = cur_dkv[0], cur_dkv[1]

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class AttnFuncWithCPAndKVA2AForTHD(torch.autograd.Function):
    """
    Attention implementation with context parallelism, which is by transmitting KV among CP ranks using AlltoAll.
    This implementation is for THD format
    """
    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            n_head,
            attention_mask,
            qkv_format,
            attn_mask_type,
            attention_dropout,
            softmax_scale,
            deterministic,
            cp_group,
            cu_seqlens_q,
            cu_seqlens_kv,
            ha_params
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        if attention_mask is not None or 'causal' in attn_mask_type:
            raise AssertionError("Currently in the implementation of Hamilton Attention, only full attention pattern"
                                 " is supported, please set 'attention_mask' as 'None', and 'attn_mask_type' as 'full'")
        if cu_seqlens_q != cu_seqlens_kv:
            raise AssertionError("cu_seqlens_q and cu_seqlens_kv must be the same for THD format.")

        path_num = ha_params.get("path_num", None)
        if not isinstance(path_num, int):
            raise TypeError(f"'path_num' must be type of 'int', but got {type(path_num).__name__}")
        permute_indices = ha_params.get("permute_index", None)
        if permute_indices is None:
            raise AssertionError("'permute_index' should be configured in Hamilton Attention.")
        restore_indices = ha_params.get("restore_index", None)
        if restore_indices is None:
            raise AssertionError("'restore_index' should be configured in Hamilton Attention.")
        out_mapping = ha_params.get("out_mapping", None)
        if not isinstance(out_mapping, list):
            raise TypeError(f"'out_mapping' must be type of 'list', but got {type(out_mapping).__name__}")
        in_mapping = ha_params.get("in_mapping", None)
        if not isinstance(in_mapping, list):
            raise TypeError(f"'in_mapping' must be type of 'list, but got {type(in_mapping).__name__}'")
        out_mapping_of_this_rank = out_mapping[rank]
        in_mapping_of_this_rank = in_mapping[rank]
        a2a_comm = A2AComm(cp_group)

        # q, k and v should be of shape: [T, N, D]
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise AssertionError(f"The q/k/v tensor should be of 3 dimensions (TND format), but got q.ndim={q.ndim}, "
                                 f"k.ndim={k.ndim}, v.ndim={v.ndim}")
        cu_seqlens_q_pad_tensor = torch.nn.functional.pad(torch.tensor(cu_seqlens_q), (1, 0), value=0)
        # actual full seq_lens of all sequences
        actual_seqlens_full = (cu_seqlens_q_pad_tensor[1:] - cu_seqlens_q_pad_tensor[:-1]).tolist()
        # current sub seq_lens of all sequences after CP split (this is the current state of q/k/v)
        cur_sub_seqlens = [x // cp_size for x in actual_seqlens_full]
        for seq_l in cur_sub_seqlens:
            if seq_l % path_num != 0:
                raise AssertionError(f"The sub sequence length: {seq_l} is not a multiple of path_num: {path_num} in "
                                     f"Hamilton Attention.")

        cu_cur_sub_seqlens_pad = torch.nn.functional.pad(torch.tensor(cur_sub_seqlens).cumsum(dim=0), (1, 0), value=0).tolist() # with padding 0
        seq_dim = 0
        cur_kv = None
        cur_kv_split = AttnFuncWithCPAndKVA2AForTHD.permute_kv_tensors(k, v, seq_dim, permute_indices, path_num)
        next_kv_split = [torch.zeros_like(cur_kv_split[p]) for p in range(path_num)]

        global_attn_outs = (None, None, None)
        for j in range(cp_size):
            if j < cp_size - 1:
                kv_send = [cur_kv_split[out_mapping_of_this_rank[i]] if out_mapping_of_this_rank[i] != -1
                           else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                kv_recv = [next_kv_split[in_mapping_of_this_rank[i]] if in_mapping_of_this_rank[i] != -1
                           else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                a2a_comm.all_to_all(kv_send, kv_recv)

            cur_kv = AttnFuncWithCPAndKVA2AForTHD.restore_kv_tensors(cur_kv_split, seq_dim, restore_indices, path_num)
            cur_k, cur_v = cur_kv[0], cur_kv[1]
            attn_outs = torch_npu.npu_fusion_attention(
                q,
                cur_k,
                cur_v,
                n_head,
                "TND",
                pse=None,
                padding_mask=None,
                atten_mask=None,
                scale=softmax_scale,
                keep_prob=1 - attention_dropout,
                actual_seq_qlen=cu_cur_sub_seqlens_pad,
                actual_seq_kvlen=cu_cur_sub_seqlens_pad,
                sparse_mode=0
            )
            global_attn_outs = general_output_update_for_ha_of_tnd_format(j, attn_outs, global_attn_outs,
                                                                          cur_sub_seqlens)

            if a2a_comm.wait():
                cur_kv_split, next_kv_split = next_kv_split, cur_kv_split

        g_attn_out, g_softmax_max, g_softmax_sum = global_attn_outs
        ctx.save_for_backward(q, cur_kv[0].clone(), cur_kv[1].clone(), g_attn_out, g_softmax_max, g_softmax_sum)
        ctx.n_head = n_head
        ctx.attention_dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.ha_params = ha_params
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = rank
        ctx.cu_cur_sub_seqlens_pad = cu_cur_sub_seqlens_pad

        return g_attn_out

    @staticmethod
    def backward(ctx, dout):
        cu_cur_sub_seqlens_pad = ctx.cu_cur_sub_seqlens_pad
        rank = ctx.cp_rank
        cp_size = ctx.cp_size
        cp_group = ctx.cp_group
        ha_params = ctx.ha_params
        softmax_scale = ctx.softmax_scale
        attention_dropout = ctx.attention_dropout
        n_head = ctx.n_head
        q, k, v, g_attn_out, g_softmax_max, g_softmax_sum = ctx.saved_tensors

        path_num = ha_params.get("path_num", None)
        out_mapping = ha_params.get("out_mapping", None)
        in_mapping = ha_params.get("in_mapping", None)
        permute_indices = ha_params.get("permute_index", None)
        restore_indices = ha_params.get("restore_index", None)
        out_mapping_of_this_rank = in_mapping[rank]
        in_mapping_of_this_rank = out_mapping[rank]
        kv_a2a_comm = A2AComm(cp_group)
        dkv_a2a_comm = A2AComm(cp_group)

        seq_dim = 0
        cur_kv_split = AttnFuncWithCPAndKVA2AForTHD.permute_kv_tensors(k, v, seq_dim, permute_indices, path_num)
        next_kv_split = [torch.zeros_like(cur_kv_split[p]) for p in range(path_num)]
        cur_dkv_split = [torch.zeros_like(cur_kv_split[p]) for p in range(path_num)]
        next_dkv_split = [torch.zeros_like(cur_kv_split[p]) for p in range(path_num)]
        cur_dkv = None

        dq = torch.zeros_like(q)
        for j in range(cp_size):
            # send kv
            if j < cp_size - 1:
                kv_send = [cur_kv_split[out_mapping_of_this_rank[i]] if out_mapping_of_this_rank[i] != -1
                           else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                kv_recv = [next_kv_split[in_mapping_of_this_rank[i]] if in_mapping_of_this_rank[i] != -1
                           else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                kv_a2a_comm.all_to_all(kv_send, kv_recv)

            cur_kv = AttnFuncWithCPAndKVA2AForTHD.restore_kv_tensors(cur_kv_split, seq_dim, restore_indices, path_num)
            cur_k, cur_v = cur_kv[0], cur_kv[1]

            attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                q,
                cur_k,
                cur_v,
                dout,
                n_head,
                "TND",
                pse=None,
                padding_mask=None,
                atten_mask=None,
                softmax_max=g_softmax_max,
                softmax_sum=g_softmax_sum,
                attention_in=g_attn_out,
                scale_value=softmax_scale,
                keep_prob=1 - attention_dropout,
                sparse_mode=0,
                actual_seq_qlen=cu_cur_sub_seqlens_pad,
                actual_seq_kvlen=cu_cur_sub_seqlens_pad
            )
            dq_step, dk_step, dv_step = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

            if j > 0 and dkv_a2a_comm.wait():
                cur_dkv_split, next_dkv_split = next_dkv_split, cur_dkv_split
            # restore dk/dv
            cur_dkv = AttnFuncWithCPAndKVA2AForTHD.restore_kv_tensors(cur_dkv_split, seq_dim, restore_indices, path_num)
            dk, dv = cur_dkv[0], cur_dkv[1]

            # update qkv grads
            dq.add_(dq_step)
            dk.add_(dk_step)
            dv.add_(dv_step)

            # permute dk/dv
            cur_dkv_split = AttnFuncWithCPAndKVA2AForTHD.permute_kv_tensors(dk, dv, seq_dim, permute_indices, path_num)

            # send dk/dv
            if j < cp_size - 1:
                dkv_send = [cur_dkv_split[out_mapping_of_this_rank[i]] if out_mapping_of_this_rank[i] != -1
                            else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                dkv_recv = [next_dkv_split[in_mapping_of_this_rank[i]] if in_mapping_of_this_rank[i] != -1
                            else torch.empty(0, dtype=q.dtype, device=q.device) for i in range(cp_size)]
                dkv_a2a_comm.all_to_all(dkv_send, dkv_recv)
            # recv kv
            if kv_a2a_comm.wait():
                cur_kv_split, next_kv_split = next_kv_split, cur_kv_split

        # recv dk/dv
        if dkv_a2a_comm.wait():
            cur_dkv_split, next_dkv_split = next_dkv_split, cur_dkv_split
            cur_dkv = AttnFuncWithCPAndKVA2AForTHD.restore_kv_tensors(cur_dkv_split, seq_dim, restore_indices, path_num)
        dk, dv = cur_dkv[0], cur_dkv[1]

        return (dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None
                )

    @classmethod
    def permute_kv_tensors(cls, k, v, seq_dim, permute_indices, path_num):
        k_permuted = k.index_select(seq_dim, permute_indices)
        v_permuted = v.index_select(seq_dim, permute_indices)

        total_seq_len = k.shape[seq_dim]
        tokens_per_path = total_seq_len // path_num
        cur_kv_split = []
        for p in range(path_num):
            start = p * tokens_per_path
            end = start + tokens_per_path
            kv_chunk = torch.stack((k_permuted[start:end], v_permuted[start:end]), dim=0)
            cur_kv_split.append(kv_chunk)
        return cur_kv_split

    @classmethod
    def restore_kv_tensors(cls, kv_split, seq_dim, restore_indices, path_num):
        all_k = [kv_split[p][0] for p in range(path_num)]
        all_v = [kv_split[p][1] for p in range(path_num)]
        k_path_major = torch.cat(all_k, dim=seq_dim)
        v_path_major = torch.cat(all_v, dim=seq_dim)

        # restore to sequence-major
        k_original = k_path_major.index_select(seq_dim, restore_indices)
        v_original = v_path_major.index_select(seq_dim, restore_indices)
        cur_kv = torch.stack((k_original, v_original), dim=0)
        return cur_kv


class A2AComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.handles = []

    def all_to_all(self, send_tensors: List[torch.Tensor], recv_tensors: List[torch.Tensor], async_op=True):
        if len(send_tensors) != len(recv_tensors):
            raise AssertionError(f"The length of send tensors and recv tensors should be equal in all_to_all, "
                                 f"but got {len(send_tensors)} and {len(recv_tensors)}")
        handle = dist.all_to_all(recv_tensors, send_tensors, group=self._process_group, async_op=async_op)
        self.handles.append(handle)

    def wait(self):
        if len(self.handles) > 0:
            for handle in self.handles:
                handle.wait()
            self.handles = []
            return True
        else:
            return False
