# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
import torch_npu
from torch import Tensor
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.training import get_args
from megatron.core import mpu, parallel_state
from mindspeed.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale
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
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import get_attention_mask
from mindspeed.utils import get_actual_seq_len
from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_init(
    self,
    config: TransformerConfig,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    attention_dropout: float = None,
):
    cp_size = config.context_parallel_size
    config.context_parallel_size = 1

    super(DotProductAttention, self).__init__(config=config)
    assert (
            self.config.context_parallel_size == 1
    ), "Context parallelism is only supported by TEDotProductAttention!"

    assert (
            self.config.window_size is None
    ), "Sliding Window Attention is only supported by TEDotProductAttention!"

    self.layer_number = max(1, layer_number)
    self.attn_mask_type = attn_mask_type
    self.attention_type = attention_type  # unused for now

    projection_size = self.config.kv_channels * self.config.num_attention_heads
    args = get_args()
    # Per attention head and per partition values.
    world_size = args.tp_x if args.tp_2d else parallel_state.get_tensor_model_parallel_world_size()
    self.hidden_size_per_partition = divide(projection_size, world_size)
    self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
    self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

    coeff = None
    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.config.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        input_in_fp16=self.config.fp16,
        input_in_bf16=self.config.bf16,
        attn_mask_type=self.attn_mask_type,
        scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
        mask_func=attention_mask_func,
        softmax_in_fp32=self.config.attention_softmax_in_fp32,
        scale=coeff,
    )

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.
    self.attention_dropout = torch.nn.Dropout(
        self.config.attention_dropout if attention_dropout is None else attention_dropout
    )

    config.context_parallel_size = cp_size

    # add pse
    args = get_args()
    self.pse = None
    self.pse_type = args.alibi_fusion_attn_type

    if args.multi_head_latent_attention:
        self.scale_mask_softmax.scale = True
        self.hidden_size_per_partition = args.num_attention_heads * args.v_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if args.rope_scaling_type is not None:
            mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
            scaling_factor = args.rope_scaling_factor

            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = 1.0 / self.softmax_scale

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
    actual_seq_len = get_actual_seq_len()

    seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    

    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

    cp_expanded_by_2d_tp = args.tp_2d and args.tp_y > 1
    if cp_expanded_by_2d_tp:
        tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
    else:
        tp_y_cp_sz = args.context_parallel_size
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
        cp_para['causal'] = args.cp_attention_mask_type == 'causal'
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

            if args.context_parallel_size > 1 and not args.tp_2d:
                cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
                cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
                cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
                cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()

            output = ringattn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask, self.attention_dropout.p,
                                           actual_seq_len, actual_seq_len)
        else:
            cp_para['scheduling_info'] = get_scheduling_info()
            output = adaptive_attn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask, self.attention_dropout.p)

    else:
        if actual_seq_len is not None: # TND
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            shape_order = 'TND'
        else: # SBH
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
                sparse_mode=args.sparse_mode,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len
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
                sparse_mode=args.sparse_mode,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len
                )[0]
        if actual_seq_len is not None:
            output = rearrange(output, '(b s) h d -> s b (h d)', s=seq_length, b=bsz)
    return output
