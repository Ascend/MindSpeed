# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
import torch_npu
from torch import Tensor
from megatron.training import get_args
from megatron.core import mpu
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.model.transformer import get_attention_mask

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['config']
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        fn(self, *args, **kwargs)
        config.context_parallel_size = cp_size

        # add pse
        args = get_args()
        self.pse = None
        self.pse_type = args.alibi_fusion_attn_type

        if self.pse_type is None:
            self.pse_type = 1 # not use pse
        elif self.pse_type == 0:
            alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(args.seq_length,
                                                    args.num_attention_heads,
                                                    args.params_dtype,
                                                    args.alibi_diagonal_opposite,
                                                    1024)
            self.pse = alibi
        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(args.num_attention_heads)

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        attention_mask = get_attention_mask()
        if get_args().use_flash_attn:
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)
        return fn(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)

    return wrapper


def dot_product_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    assert packed_seq_params is None
    args = get_args()

    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key = key.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
        )
        value = value.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
        )

    seq_length, _, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    
    query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]

    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
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
        cp_para['causal'] = args.cp_attention_mask_type == 'causal'
        cp_para['cp_group'] = cp_group
        cp_para['cp_size'] = cp_size
        cp_para['rank'] = rank
        cp_para['cp_global_ranks'] = cp_global_ranks
        cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None
        cp_para['pse'] = self.pse
        cp_para['pse_type'] = self.pse_type

        cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
        cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
        cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
        cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
        cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()

        output = ringattn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask, self.attention_dropout.p)

    else:
        if args.use_fusion_attn_v2:
            output = npu_fusion_attention(
                query, key, value, n_head, 'SBH',
                pse=self.pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pse_type=self.pse_type,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]
        else:
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, 'SBH',
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
                )[0]
    return output
