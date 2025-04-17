# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

from torch import Tensor
from megatron.core.transformer.dot_product_attention import DotProductAttention as MegatronDotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams

from mindspeed.core.transformer.flash_attention.alibi.dot_product_attention import DotProductAttentionImpl


class MindSpeedDotProductAttention(DotProductAttentionImpl, MegatronDotProductAttention):

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
    ):
        MegatronDotProductAttention.__init__(
            self,
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type
        )
        
        # add pse
        DotProductAttentionImpl.__init__(self)

