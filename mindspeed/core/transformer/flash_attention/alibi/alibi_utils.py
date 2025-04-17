# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import torch
from megatron.core import parallel_state


def get_slopes(n):
    """
    Generate ALiBi slopes for n attention heads.
    The slopes are computed based on the number of heads and follow a power-of-2 pattern.

    Args:
        n (int): Number of attention heads.

    Returns:
        List[float]: A list of slopes for each attention head.
    """

    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2) +
            get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
        )


def _get_inverted_mask(attention_mask, alibi):
    inverted_mask = attention_mask.to(alibi.dtype)
    inverted_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), float("-inf")
    )
    return inverted_mask.to(alibi.device) + alibi.unsqueeze(0)


def _build_alibi_tensor(
    max_seq_len,
    num_attention_heads,
    square_alibi_mask,
    fill_neg_inf
):
    def _fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def _buffered_future_mask(maxpos, alibi, attn_heads):
        _future_mask = torch.triu(
            _fill_with_neg_inf(torch.zeros([maxpos, maxpos])),
            1
        )
        _future_mask = _future_mask.unsqueeze(0) + alibi
        return _future_mask[:attn_heads, :maxpos, :maxpos]

    slopes = torch.Tensor(get_slopes(num_attention_heads))
    if square_alibi_mask:
        position_point = torch.arange(max_seq_len) - max_seq_len + 1
        position_point = (
            position_point.unsqueeze(0).unsqueeze(0).expand(
                num_attention_heads, max_seq_len, -1
            )
        )
        diag = torch.diag(position_point[0])
        position_point = (
            position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        )
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    else:
        alibi = (
            slopes.unsqueeze(1).unsqueeze(1) *
            torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
                num_attention_heads, -1, -1
            )
        )

    # Select the part of the tensor that corresponds to our tensor parallel index.
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_index = parallel_state.get_tensor_model_parallel_rank()
    alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

    if fill_neg_inf:
        return _buffered_future_mask(max_seq_len, alibi, num_attention_heads)

    return alibi
