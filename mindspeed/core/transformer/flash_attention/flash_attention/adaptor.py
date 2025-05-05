# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import Optional

from torch import Tensor
import torch_npu

from megatron.core.transformer.dot_product_attention import DotProductAttention as MegatronDotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_forward_impl(
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

    scale = (
        1.0 / math.sqrt(self.hidden_size_per_attention_head) 
        if self.scale_mask_softmax.scale is None 
        else self.softmax_scale
    )

    if packed_seq_params is not None: # TND
        actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
        actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
        query, key, value = (
            [
                rearrange(x, 's b h d -> (b s) h d')
                for x in [query, key, value]
            ]
        )
        shape_order = 'TND'
    else: # SBH
        actual_seq_qlen = None
        actual_seq_kvlen = None
        query, key, value = (
            [
                rearrange(x, 's b h d -> s b (h d)')
                for x in [query, key, value]
            ]
        )
        shape_order = 'SBH'

    output = torch_npu.npu_fusion_attention(
        query, key, value,
        n_head,
        shape_order,
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=scale,
        pre_tockens=self.config.pre_tockens,
        next_tockens=self.config.next_tockens,
        keep_prob=1 - self.attention_dropout.p,
        inner_precise=0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen
    )[0]

    if packed_seq_params is not None:
        output = (
            rearrange(
                output,
                '(b s) h d -> s b (h d)',
                s=seq_length, b=bsz
            )
        )
        
    return output
