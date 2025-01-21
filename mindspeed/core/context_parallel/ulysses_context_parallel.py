# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple
import torch
import torch_npu
from torch import Tensor
from torch.nn import Module
from megatron.training import get_args
import mindspeed
from mindspeed.core.tensor_parallel.comm_group_api import CollectiveCommIntf
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_first_dim, sync_reduce_scatter_along_first_dim

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class UlyssesCollectiveComm(CollectiveCommIntf):
    group = None

    def __init__(self, group, name="ulysses"):
        super().__init__(name)
        UlyssesCollectiveComm.group = group

    @classmethod
    def get_comm_rank(cls):
        return torch.distributed.get_rank(group=cls.group)

    @classmethod
    def get_comm_group_world_size(cls):
        return torch.distributed.get_world_size(group=cls.group)

    @classmethod
    def get_comm_group(cls):
        return cls.group


def single_all_to_all(input_, scatter_idx, gather_idx, group):
    seq_world_size = torch.distributed.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input_.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input_.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    torch.distributed.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    # e.g., [cp, s/cp, b, n/cp, d] -> [s/cp, b, cp, n/cp, d]
    if scatter_idx < 2:
        output = output.transpose(0, 1).transpose(1, 2).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + [inp_shape[gather_idx] * seq_world_size, ] + inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: torch.distributed.ProcessGroup, input_: Tensor, scatter_idx: int,
                gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input_, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class UlyssesContextAttention(torch.nn.Module):
    """Initialization.
    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """
    def __init__(
            self,
            local_attention: Module,
            sequence_process_group: torch.distributed.ProcessGroup,
            scatter_idx: int = 2,
            gather_idx: int = 0,
    ) -> None:
        super(UlyssesContextAttention, self).__init__()
        self.local_attn = local_attention
        self.local_attn.ulysses_comm_para = dict()
        self.local_attn.ulysses_comm_para['spg'] = sequence_process_group
        self.local_attn.ulysses_comm_para['scatter_idx'] = scatter_idx
        self.local_attn.ulysses_comm_para['gather_idx'] = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        use_custom_ulysses_backward = (
            get_args().context_parallel_size > 1 and
            get_args().context_parallel_algo == "ulysses_cp_algo" and
            not get_args().use_legacy_models
        )
        if use_custom_ulysses_backward:
            output = self.local_attn(query, key, value, *args, **kwargs)
        else:
            spg = self.local_attn.ulysses_comm_para.get('spg')
            scatter_idx = self.local_attn.ulysses_comm_para.get('scatter_idx')
            gather_idx = self.local_attn.ulysses_comm_para.get('gather_idx')
            seq_world_size = torch.distributed.get_world_size(spg)
            if seq_world_size > key.shape[scatter_idx] and query.shape[scatter_idx] % key.shape[scatter_idx] == 0:
                key = key.repeat_interleave(query.shape[scatter_idx] // key.shape[scatter_idx], dim=scatter_idx)
                value = value.repeat_interleave(query.shape[scatter_idx] // value.shape[scatter_idx], dim=scatter_idx)

            # in shape : e.g.,  [s/p:h:]
            query_layer = _SeqAllToAll.apply(spg, query, scatter_idx, gather_idx)
            key_layer = _SeqAllToAll.apply(spg, key, scatter_idx, gather_idx)
            value_layer = _SeqAllToAll.apply(spg, value, scatter_idx, gather_idx)

            # out shape : e.g., [s:h/p:]
            context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)

            output = _SeqAllToAll.apply(spg, context_layer, gather_idx, scatter_idx)

        # out e.g., [s/p::h]
        return output


class AttnQKVReshape:
    """Ulysses Attention Reshape QKV Implementation"""

    def __init__(self, attn_para):
        self.attn_para = attn_para

    def reshape_forward(self, query, key, value):
        """
        Implements of qkv reshape in forward of ulysses attention

        Args:
            query (Tensor): query input to the attention layer with shape [s, b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h, d]

        Returns:
            query (Tensor): query input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            attn_para (Dict): the parameters used in attention computation
        """
        # q, k, v: [s, b, h, d]

        # attention parameters
        packed_seq_params = self.attn_para.get('packed_seq_params')
        if packed_seq_params is None:
            actual_seq_len = mindspeed.utils.get_actual_seq_len()
            seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
        else:
            actual_seq_len = tuple(packed_seq_params.cu_seqlens_q[1:].cpu().numpy().tolist())
            seq_length, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2]

        # reshape [s, b, h, d] to SBH([s, b, h*d]) or TND([s*b, h, d])
        if actual_seq_len is not None: # TND
            if packed_seq_params is None:
                query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            shape_order = 'TND'
        else: # SBH
            query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            shape_order = 'SBH'

        self.attn_para['n_head'] = n_head
        self.attn_para['shape_order'] = shape_order
        self.attn_para['seq_length'] = seq_length
        self.attn_para['actual_seq_qlen'] = actual_seq_len
        self.attn_para['actual_seq_kvlen'] = actual_seq_len

        return query, key, value, self.attn_para

    def reshape_backward(self, dq, dk, dv):
        """
        Implements of qkv reshape in backward of ulysses attention

        Args:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]

        Returns:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h, d]
        """
        # dq, dk, dv: [s, b, h*d] or [s*b, h, d]

        # attention parameters
        packed_seq_params = self.attn_para.get('packed_seq_params')
        actual_seq_len = self.attn_para.get('actual_seq_qlen')
        seq_length = self.attn_para.get('seq_length')
        n_head = self.attn_para.get('n_head')

        # reshape SBH([s, b, h*d]) or TND([s*b, h, d]) back to [s, b, h, d]
        if actual_seq_len is not None:  # TND
            if packed_seq_params is None:
                s, b = seq_length, dq.shape[0] // seq_length
                dq, dk, dv = [rearrange(x, '(b s) h d -> s b h d', s=s, b=b) for x in [dq, dk, dv]]
        else:  # SBH
            h, d = n_head, dq.shape[2] // n_head
            dq, dk, dv = [rearrange(x, 's b (h d) -> s b h d', h=h, d=d) for x in [dq, dk, dv]]

        return dq, dk, dv


class RepeatAll2AllComm:
    """Ulysses Attention Repeat All2All Communication Implementation"""

    def __init__(self, ulysses_comm_para, attn_para):
        self.ulysses_comm_para = ulysses_comm_para
        self.attn_para = attn_para
        self.qkv_reshape = AttnQKVReshape(attn_para)

    def comm_forward(self, query, key, value):
        """
        Implements of Repeat-All2All communication in forward of ulysses attention

        Args:
            query (Tensor): query input to the attention layer with shape [s, b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h, d]

        Returns:
            query (Tensor): query input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            attn_para (Dict): the parameters used in attention computation
        """
        # q, k, v: [s, b, h, d]

        # communication parameters
        spg = self.ulysses_comm_para.get('spg')
        scatter_idx = self.ulysses_comm_para.get('scatter_idx')
        gather_idx = self.ulysses_comm_para.get('gather_idx')
        cache_policy = self.ulysses_comm_para.get('cache_policy')

        # repeat parameters
        seq_world_size = torch.distributed.get_world_size(spg)
        do_repeat = seq_world_size > key.shape[scatter_idx] and query.shape[scatter_idx] % key.shape[scatter_idx] == 0
        self.ulysses_comm_para['do_repeat'] = do_repeat
        self.ulysses_comm_para['repeat_num'] = query.shape[scatter_idx] // key.shape[scatter_idx]

        # if forward repeat, [s, b, h, d] -> [s, b, h*cp, d]
        if do_repeat:
            key = key.repeat_interleave(query.shape[scatter_idx] // key.shape[scatter_idx], dim=scatter_idx)
            value = value.repeat_interleave(query.shape[scatter_idx] // value.shape[scatter_idx], dim=scatter_idx)
        elif cache_policy is not None:
            raise AssertionError(
                'KV Cache dose not suggest to use when key and value do not repeat'
            )

        # all2all communication forward, [s, b, h, d] -> [s*cp, b, h//cp, d]
        query = single_all_to_all(query, scatter_idx, gather_idx, spg)
        key = single_all_to_all(key, scatter_idx, gather_idx, spg)
        value = single_all_to_all(value, scatter_idx, gather_idx, spg)

        # reshape [s, b, h, d] to SBH([s, b, h*d]) or TND([s*b, h, d])
        query, key, value, self.attn_para = self.qkv_reshape.reshape_forward(query, key, value)

        return query, key, value, self.attn_para

    def comm_backward(self, dq, dk, dv):
        """
        Implements of Repeat-All2All communication in backward of ulysses attention

        Args:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]

        Returns:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h, d]
        """
        # dq, dk, dv: SBH([s, b, h*d]) or TND([s*b, h, d])

        # reshape SBH([s, b, h*d]) or TND([s*b, h, d]) back to [s, b, h, d]
        dq, dk, dv = self.qkv_reshape.reshape_backward(dq, dk, dv)

        # communication parameters
        spg = self.ulysses_comm_para.get('spg')
        scatter_idx = self.ulysses_comm_para.get('scatter_idx')
        gather_idx = self.ulysses_comm_para.get('gather_idx')
        do_repeat = self.ulysses_comm_para.get('do_repeat')
        repeat_num = self.ulysses_comm_para.get('repeat_num')

        # all2all communication backward, [s, b, h, d] -> [s//cp, b, h*cp, d]
        dq = single_all_to_all(dq, gather_idx, scatter_idx, spg)
        dk = single_all_to_all(dk, gather_idx, scatter_idx, spg)
        dv = single_all_to_all(dv, gather_idx, scatter_idx, spg)

        # if backward repeat, [s, b, h, d] -> [s, b, h//cp, d]
        if do_repeat:
            dk = dk.view(
                *dk.shape[:scatter_idx], dk.shape[scatter_idx] // repeat_num, repeat_num, *dk.shape[scatter_idx + 1:]
            ).sum(dim=scatter_idx + 1)
            dv = dv.view(
                *dv.shape[:scatter_idx], dv.shape[scatter_idx] // repeat_num, repeat_num, *dv.shape[scatter_idx + 1:]
            ).sum(dim=scatter_idx + 1)

        return dq, dk, dv

    def recomm_backward(self, input_tensor):
        """
        Implements of Repeat-All2All re-communication in backward of ulysses attention

        Args:
            input_tensor (Tensor): key or value input of the attention layer with shape [s, b, h, d]

        Returns:
            output (Tensor): key or value input of the attention layer with shape [s, b, h*d] or [s*b, h, d]
        """
        # k, v: [s, b, h, d]

        # communication parameters
        spg = self.ulysses_comm_para.get('spg')
        scatter_idx = self.ulysses_comm_para.get('scatter_idx')
        gather_idx = self.ulysses_comm_para.get('gather_idx')
        do_repeat = self.ulysses_comm_para.get('do_repeat')
        repeat_num = self.ulysses_comm_para.get('repeat_num')

        # attention parameters
        packed_seq_params = self.attn_para.get('packed_seq_params')
        actual_seq_len = self.attn_para.get('actual_seq_qlen')

        # if repeat, [s, b, h, d] -> [s, b, h*cp, d]
        if do_repeat:
            input_tensor = input_tensor.repeat_interleave(repeat_num, dim=scatter_idx)

        # all2all re-communication, [s, b, h, d] -> [s*cp, b, h//cp, d]
        output = single_all_to_all(input_tensor, scatter_idx, gather_idx, spg)

        # reshape [s, b, h, d] to SBH([s, b, h*d]) or TND([s*b, h, d])
        if actual_seq_len is not None:  # TND
            if packed_seq_params is None:
                output = rearrange(output, 's b h d -> (b s) h d')
        else:  # SBH
            output = rearrange(output, 's b h d -> s b (h d)')

        return output


class AllGatherComm:
    """Ulysses Attention AllGather KV + All2All Q Communication Implementation"""

    def __init__(self, ulysses_comm_para, attn_para):
        self.ulysses_comm_para = ulysses_comm_para
        self.attn_para = attn_para
        self.qkv_reshape = AttnQKVReshape(attn_para)
        spg = self.ulysses_comm_para.get('spg')
        self.ulysses_collective_comm = UlyssesCollectiveComm(spg)

    def comm_forward(self, query, key, value):
        """
        Implements of AllGather KV + All2All Q communication in forward of ulysses attention

        Args:
            query (Tensor): query input to the attention layer with shape [s, b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h, d]

        Returns:
            query (Tensor): query input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h*d] or [s*b, h, d]
            attn_para (Dict): the parameters used in attention computation
        """
        # q, k, v: [s, b, h, d]

        # communication parameters
        spg = self.ulysses_comm_para.get('spg')
        scatter_idx = self.ulysses_comm_para.get('scatter_idx')
        gather_idx = self.ulysses_comm_para.get('gather_idx')

        # query all2all communication forward, [s, b, h, d] -> [s*cp, b, h//cp, d]
        query = single_all_to_all(query, scatter_idx, gather_idx, spg)

        # key and value allgather communication forward, [s, b, h, d] -> [s*cp, b, h, d]
        key = sync_gather_along_first_dim(key, self.ulysses_collective_comm)
        value = sync_gather_along_first_dim(value, self.ulysses_collective_comm)

        # reshape [s, b, h, d] to SBH([s, b, h*d]) or TND([s*b, h, d])
        query, key, value, self.attn_para = self.qkv_reshape.reshape_forward(query, key, value)

        return query, key, value, self.attn_para

    def comm_backward(self, dq, dk, dv):
        """
        Implements of AllGather KV + All2All Q communication in backward of ulysses attention

        Args:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h*d] or [s*b, h, d]

        Returns:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h, d]
        """
        # dq, dk, dv: SBH([s, b, h*d]) or TND([s*b, h, d])

        # reshape SBH([s, b, h*d]) or TND([s*b, h, d]) back to [s, b, h, d]
        dq, dk, dv = self.qkv_reshape.reshape_backward(dq, dk, dv)

        # communication parameters
        spg = self.ulysses_comm_para.get('spg')
        scatter_idx = self.ulysses_comm_para.get('scatter_idx')
        gather_idx = self.ulysses_comm_para.get('gather_idx')

        # query all2all communication backward, [s, b, h, d] -> [s//cp, b, h*cp, d]
        dq = single_all_to_all(dq, gather_idx, scatter_idx, spg)

        # key and value allgather communication backward, [s, b, h, d] -> [s//cp, b, h, d]
        dk = sync_reduce_scatter_along_first_dim(dk, self.ulysses_collective_comm)
        dv = sync_reduce_scatter_along_first_dim(dv, self.ulysses_collective_comm)

        return dq, dk, dv

    def recomm_backward(self, input_tensor):
        """
        Implements of AllGather KV + All2All Q re-communication in backward of ulysses attention

        Args:
            input_tensor (Tensor): key or value input of the attention layer with shape [s, b, h, d]

        Returns:
            output (Tensor): key or value input of the attention layer with shape [s, b, h*d] or [s*b, h, d]
        """
        # k, v: [s, b, h, d]

        # attention parameters
        packed_seq_params = self.attn_para.get('packed_seq_params')
        actual_seq_len = self.attn_para.get('actual_seq_qlen')

        # allgather re-communication, [s, b, h, d] -> [s*cp, b, h, d]
        output = sync_gather_along_first_dim(input_tensor, self.ulysses_collective_comm)

        # reshape [s, b, h, d] to SBH([s, b, h*d]) or TND([s*b, h, d])
        if actual_seq_len is not None:  # TND
            if packed_seq_params is None:
                output = rearrange(output, 's b h d -> (b s) h d')
        else:  # SBH
            output = rearrange(output, 's b h d -> s b (h d)')

        return output


class UlyssesAttnWithKVCache(torch.autograd.Function):
    """Ulysses Attention With KV Cache Implementation"""

    @staticmethod
    def forward(ctx, query, key, value, attn_para, ulysses_comm_para) -> Tensor:
        """
        Implements of Ulysses Attention With KV Cache forward

        Args:
            query (Tensor): query input to the attention layer with shape [s, b, h, d]
            key (Tensor): key input to the attention layer with shape [s, b, h, d]
            value (Tensor): value input to the attention layer with shape [s, b, h, d]

        Returns:
            output (Tensor): ulysses attention output with shape [s, b, h*d] or [s*b, h, d]
        """
        # q, k, v: [s, b, h, d]

        # communication parameters
        spg = ulysses_comm_para.get('spg')
        scatter_idx = ulysses_comm_para.get('scatter_idx')
        gather_idx = ulysses_comm_para.get('gather_idx')
        cache_policy = ulysses_comm_para.get('cache_policy')
        use_ulysses_allgather_kv = ulysses_comm_para.get('use_ulysses_allgather_kv')

        # repeat-all2all or allgather kv + all2all q
        if use_ulysses_allgather_kv:
            if key.shape[2] != 1:
                raise AssertionError(
                    'When either the head number of key or value is not equal to 1, '
                    'use all2all communication to get better performance.'
                )
            # allgather kv + all2all q communication forward
            ulysses_comm = AllGatherComm(ulysses_comm_para, attn_para)
        else:
            # repeat-all2all communication forward
            ulysses_comm = RepeatAll2AllComm(ulysses_comm_para, attn_para)

        # communication forward
        q, k, v = query.clone(), key.clone(), value.clone()
        q, k, v, attn_para = ulysses_comm.comm_forward(q, k, v)

        # attention parameters
        packed_seq_params = attn_para.get('packed_seq_params')
        attention_mask = attn_para.get('attention_mask')
        scale = attn_para.get('scale')
        pre_tokens = attn_para.get('pre_tokens')
        next_tokens = attn_para.get('next_tokens')
        keep_prob = attn_para.get('keep_prob')
        sparse_mode = attn_para.get('sparse_mode')
        n_head = attn_para.get('n_head')
        shape_order = attn_para.get('shape_order')
        actual_seq_len = attn_para.get('actual_seq_qlen')
        actual_seq_kvlen = attn_para.get('actual_seq_kvlen')
        seq_length = attn_para.get('seq_length')

        # kv cache
        if cache_policy == "full":
            k_cache, v_cache = key.clone(), value.clone()
        elif cache_policy == "half":
            k_cache, v_cache = key.clone(), v.clone()
        else:
            k_cache, v_cache = k.clone(), v.clone()

        # attention forward
        res = torch_npu.npu_fusion_attention(
            q, k, v, n_head, shape_order,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=scale,
            pre_tockens=pre_tokens,
            next_tockens=next_tokens,
            keep_prob=keep_prob,
            inner_precise=0,
            sparse_mode=sparse_mode,
            actual_seq_qlen=actual_seq_len,
            actual_seq_kvlen=actual_seq_kvlen
        )

        attn_out, softmax_max, softmax_sum = res[0], res[1], res[2]

        # if TND, reshape TND([b*s, h, d]) to SBH([s, b, h*d])
        if actual_seq_len is not None and packed_seq_params is None:
            s, b = seq_length, attn_out.shape[0] // seq_length
            attn_out = rearrange(attn_out, '(b s) h d -> s b (h d)', s=s, b=b)

        # output all2all communication forward
        output = single_all_to_all(attn_out, gather_idx, scatter_idx, spg)

        ctx.save_for_backward(q, k_cache, v_cache, attn_out, softmax_max, softmax_sum, attention_mask)
        ctx.ulysses_comm = ulysses_comm
        ctx.ulysses_comm_para = ulysses_comm_para
        ctx.attn_para = attn_para

        return output

    @staticmethod
    def backward(ctx, dout):
        """
        Implements of Ulysses Attention With KV Cache backward

        Args:
            dout (Tensor): the attention layer output grad with shape [s, b, h*d] or [s*b, h, d]

        Returns:
            dq (Tensor): query grad output of the attention layer with shape [s, b, h, d]
            dk (Tensor): key grad output of the attention layer with shape [s, b, h, d]
            dv (Tensor): value grad output of the attention layer with shape [s, b, h, d]
        """
        # input, attention output grad: [s, b, h*d] or [s*b, h, d]

        # get forward parameters
        query, k_cache, v_cache, attn_out, softmax_max, softmax_sum, attention_mask = ctx.saved_tensors
        ulysses_comm = ctx.ulysses_comm
        ulysses_comm_para = ctx.ulysses_comm_para
        attn_para = ctx.attn_para

        # communication parameters
        spg = ulysses_comm_para.get('spg')
        scatter_idx = ulysses_comm_para.get('scatter_idx')
        gather_idx = ulysses_comm_para.get('gather_idx')
        cache_policy = ulysses_comm_para.get('cache_policy')

        # attention parameters
        packed_seq_params = attn_para.get('packed_seq_params')
        attention_mask = attn_para.get('attention_mask')
        scale = attn_para.get('scale')
        pre_tokens = attn_para.get('pre_tokens')
        next_tokens = attn_para.get('next_tokens')
        keep_prob = attn_para.get('keep_prob')
        sparse_mode = attn_para.get('sparse_mode')
        n_head = attn_para.get('n_head')
        shape_order = attn_para.get('shape_order')
        actual_seq_len = attn_para.get('actual_seq_qlen')
        actual_seq_kvlen = attn_para.get('actual_seq_kvlen')

        # output all2all communication backward
        dout = single_all_to_all(dout, scatter_idx, gather_idx, spg)

        # if TND, reshape SBH([s, b, h*d]) to TND([b*s, h, d])
        if actual_seq_len is not None and packed_seq_params is None:
            h, d = n_head, dout.shape[2] // n_head
            dout = rearrange(dout, 's b (h d) -> (b s) h d', h=h, d=d)
            attn_out = rearrange(attn_out, 's b (h d) -> (b s) h d', h=h, d=d)

        # kv cache re-communication
        if cache_policy == "full":
            key = ulysses_comm.recomm_backward(k_cache)
            value = ulysses_comm.recomm_backward(v_cache)
        elif cache_policy == "half":
            key = ulysses_comm.recomm_backward(k_cache)
            value = v_cache
        else:
            key = k_cache
            value = v_cache

        # attention backward
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            query, key, value, dout, n_head,
            shape_order,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attn_out,
            scale_value=scale,
            pre_tockens=pre_tokens,
            next_tockens=next_tokens,
            sparse_mode=sparse_mode,
            keep_prob=keep_prob,
            actual_seq_qlen=actual_seq_len,
            actual_seq_kvlen=actual_seq_kvlen
        )

        dq, dk, dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

        dq, dk, dv = ulysses_comm.comm_backward(dq, dk, dv)

        return dq, dk, dv, None, None


def ulyssesattn_context_parallel(query, key, value, attn_para, ulysses_comm_para):
    out = UlyssesAttnWithKVCache.apply(query, key, value, attn_para, ulysses_comm_para)
    return out
