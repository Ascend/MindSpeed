# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import math
from functools import wraps

import torch_npu
from torch import Tensor
from megatron.core.transformer.enums import AttnMaskType
from megatron.training import get_args
from megatron.core import mpu, parallel_state
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import get_attention_mask
from mindspeed.utils import get_actual_seq_len
from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, attention_bias, packed_seq_params):
        if attention_mask is None and self.attn_mask_type == AttnMaskType.causal:
            if not getattr(self.config, 'is_llava', False):
                attention_mask = get_attention_mask()
        if get_args().use_flash_attn:
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias, packed_seq_params)
        return fn(self, query, key, value, attention_mask, attn_mask_type, attention_bias, packed_seq_params)

    return wrapper


def dot_product_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        attention_bias,
        packed_seq_params,
):
    assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

    args = get_args()
    if packed_seq_params is None:
        actual_seq_len = get_actual_seq_len()
        seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    else:
        actual_seq_len = tuple(packed_seq_params.cu_seqlens_q[1:].cpu().numpy().tolist())
        seq_length, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2]
    sparse_mode = args.sparse_mode
    
    if attn_mask_type == AttnMaskType.no_mask:
        sparse_mode = 0  # default mask

    scale = 1.0 / math.sqrt(
        self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

    cp_expanded_by_2d_tp = args.tp_2d and args.tp_y > 1
    if cp_expanded_by_2d_tp:
        tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
    else:
        tp_y_cp_sz = self.config.context_parallel_size

    if (self.config.context_parallel_size > 1 and args.context_parallel_algo == "ulysses_cp_algo"
            and args.context_parallel_kv_cache_policy):
        self.ulysses_comm_para['cache_policy'] = get_cache_policy(
            self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
        )
        self.ulysses_comm_para['use_ulysses_allgather_kv'] = args.use_ulysses_allgather_kv

        attn_para = dict()
        attn_para['packed_seq_params'] = packed_seq_params
        attn_para['attention_mask'] = attention_mask
        attn_para['scale'] = scale
        attn_para['pre_tokens'] = args.pre_tockens
        attn_para['next_tokens'] = args.next_tockens
        attn_para['keep_prob'] = 1 - self.attention_dropout.p
        attn_para['sparse_mode'] = sparse_mode
        output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)

        return output

    if tp_y_cp_sz > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                                         'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
        in_hybrid_mode = False
        if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
            in_hybrid_mode = True
 
        if not in_hybrid_mode:
            if cp_expanded_by_2d_tp:
                tp_y_cp = TensorParallelYUnionCP()
                cp_group = tp_y_cp.group
                cp_size = tp_y_cp.get_parallel_group_world_size()
                rank = tp_y_cp.get_parallel_rank()
                cp_global_ranks = tp_y_cp.global_ranks
            else:
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
        cp_para['megatron_cp_in_bnsd'] = self.config.megatron_cp_in_bnsd
        cp_para['causal'] = args.attention_mask_type == 'causal'
        cp_para['cp_group'] = cp_group
        cp_para['cp_size'] = cp_size
        cp_para['rank'] = rank

        query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
        if args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
            cp_para['cp_global_ranks'] = cp_global_ranks
            if args.use_cp_send_recv_overlap:
                if cp_expanded_by_2d_tp:
                    cp_para['cp_group_for_send_recv_overlap'] = tp_y_cp.overlap_group
                else:
                    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap()
            else:
                cp_para['cp_group_for_send_recv_overlap'] = None
            cp_para['pse'] = self.pse
            cp_para['pse_type'] = self.pse_type

            if self.config.context_parallel_size > 1 and not args.tp_2d:
                cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
                cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
                cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
                cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
                cp_para['cache_policy'] = get_cache_policy(
                    self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
                )

            output = ringattn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask, self.attention_dropout.p,
                                           packed_seq_params)
        else:
            cp_para['scheduling_info'] = get_scheduling_info()
            output = adaptive_attn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask, self.attention_dropout.p)

    else:
        if packed_seq_params is not None: # TND
            cp_size = mpu.get_context_parallel_world_size()
            actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            shape_order = 'TND'
        else: # SBH
            actual_seq_qlen = None
            actual_seq_kvlen = None
            query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            shape_order = 'SBH'
        if args.use_fusion_attn_v2:
            output = npu_fusion_attention(
                query, key, value, n_head, shape_order,
                pse=self.pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pse_type=self.pse_type,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=sparse_mode,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen
            )[0]
        else:
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, shape_order,
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=sparse_mode,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen
                )[0]
        if packed_seq_params is not None:
            output = rearrange(output, '(b s) h d -> s b (h d)', s=seq_len, b=bsz)
    return output
