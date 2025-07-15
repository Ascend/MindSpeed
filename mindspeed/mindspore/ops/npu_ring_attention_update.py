# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
__all__ = ["npu_ring_attention_update"]

import torch
from mindspore.ops.auto_generate import ring_attention_update


def npu_ring_attention_update(
        prev_attn_out: torch.Tensor,
        prev_softmax_max: torch.Tensor,
        prev_softmax_sum: torch.Tensor,
        cur_attn_out: torch.Tensor,
        cur_softmax_max: torch.Tensor,
        cur_softmax_sum: torch.Tensor,
        actual_seq_qlen: torch.Tensor = None,
        layout: str = "SBH",
):
    return ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)
