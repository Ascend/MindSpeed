# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import Optional

from torch import Tensor

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams

from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.core.transformer.flash_attention.alibi.alibi import AlibiForFusionAttnSingleton

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class DotProductAttentionImpl():
    """
    Implementation of dot product attention with ALiBi support.
    """

    def __init__(self):
        # add pse
        self.pse = None
        self.pse_type = self.config.alibi_fusion_attn_type

        if self.pse_type is None:
            self.pse_type = 1 # not use pse
        elif self.pse_type == 0:
            alibi = (
                AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(
                    self.config.seq_length,
                    self.config.num_attention_heads,
                    self.config.params_dtype,
                    self.config.alibi_diagonal_opposite,
                    1024
                )
            )
            self.pse = alibi
        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = (
                AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(
                    self.config.num_attention_heads
                )
            )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        assert attention_bias is None, \
            "Attention bias is not supported for DotProductAttention."

        if packed_seq_params is None:
            seq_length, bsz, n_head, head_dim = (
                query.shape[0], query.shape[1], query.shape[2], query.shape[3]
            )
        else:
            seq_length, n_head, head_dim = (
                query.shape[0], query.shape[1], query.shape[2]
            )

        sparse_mode = self.config.sparse_mode
        if attn_mask_type == AttnMaskType.no_mask:
            sparse_mode = 0  # default mask

        scale = self.softmax_scale
        
        if packed_seq_params is not None: # TND
            actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
            shape_order = 'TND'
        else: # SBH
            actual_seq_qlen = None
            actual_seq_kvlen = None
            query, key, value = (
                [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            )
            shape_order = 'SBH'

        output = npu_fusion_attention(
            query, key, value, n_head, shape_order,
            pse=self.pse,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=scale,
            pse_type=self.pse_type,
            pre_tokens=self.config.pre_tockens,
            next_tokens=self.config.next_tockens,
            keep_prob=1 - self.attention_dropout.p,
            inner_precise=0,
            sparse_mode=sparse_mode,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen
        )[0]

        return output