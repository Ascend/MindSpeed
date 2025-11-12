# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
Rotary Embedding Wrapper for Seq1F1B

This module provides a wrapper for rotary position embedding to handle sequence splits
in Seq1F1B scheduling. It adjusts position offsets and sequence lengths based
on the current span information and MLA configuration.
"""

from functools import wraps
from torch import Tensor
from megatron.training import get_args


def rotary_embedding_forward_wrapper(forward_func):
    @wraps(forward_func)
    def rotary_embedding_forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        global_args = get_args()
        if global_args.seq1f1b_splits > 1:
            offset = global_args.span_info.span_start
            if global_args.multi_latent_attention:
                max_seq_len += offset
                offset = 0
        rotary_pos_emb = forward_func(self, max_seq_len, offset, packed_seq)
        return rotary_pos_emb
    return rotary_embedding_forward