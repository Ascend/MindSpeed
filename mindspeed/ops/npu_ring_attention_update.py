# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch

from mindspeed.op_builder import RingAttentionUpdateOpBuilder

__all__ = ["npu_ring_attention_update"]


op_builder = RingAttentionUpdateOpBuilder()


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
    ops = op_builder.load()
    return ops.npu_ring_attention_update(
        prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum,
        actual_seq_qlen, layout)
