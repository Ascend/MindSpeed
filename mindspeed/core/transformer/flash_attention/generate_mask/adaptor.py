# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional
from functools import wraps

from torch import Tensor

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.transformer.flash_attention.generate_mask.generate_mask import get_attention_mask


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        if (
            attention_mask is None and 
            self.attn_mask_type == AttnMaskType.causal
        ) and not getattr(self.config, 'is_llava', False):
            attention_mask = get_attention_mask(self.config)
        return fn(
            self, query, key, value, 
            attention_mask, attn_mask_type, attention_bias, packed_seq_params
            )
    return wrapper