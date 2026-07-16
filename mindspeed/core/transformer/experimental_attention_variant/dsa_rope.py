# Copyright (c) 2024, Huawei Technologies Co., Ltd.
#
# DSA RoPE optimization (P4).
#
# Provides an alternative implementation of RoPE in the complex domain,
# which can be more efficient on NPU compared to the standard
# apply_rotary_pos_emb implementation.
#
# Reference: MindSpeed 0.16 core_r0.16.0 dsa_matrix_naive.py

import torch


def apply_rope_in_complex(x, rotary_pos_emb, mscale=1.0):
    """Apply RoPE in complex domain.

    Alternative to standard apply_rotary_pos_emb that performs the
    rotation in the complex number domain, which can be more efficient
    on NPU hardware.

    Args:
        x: [seqlen, batch, *, dim] Input tensor (only last dim is rotated)
        rotary_pos_emb: [1, seqlen, 1, dim] RoPE frequencies
        mscale: float, scaling factor for YaRN

    Returns:
        Rotated tensor with same shape as input
    """
    # Split into two halves for complex multiplication
    dim = x.size(-1)
    half_dim = dim // 2
    x_part1, x_part2 = x[..., :half_dim], x[..., half_dim:]

    # Megatron 0.17 RotaryEmbedding returns freqs with shape [seq, 1, 1, dim].
    freqs = rotary_pos_emb[..., :half_dim]
    while freqs.dim() > x.dim():
        squeezed = False
        for dim in range(1, freqs.dim() - 1):
            if freqs.size(dim) == 1:
                freqs = freqs.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            break
    cos = (torch.cos(freqs) * mscale).to(x.dtype)
    sin = (torch.sin(freqs) * mscale).to(x.dtype)

    # Ensure shapes broadcast correctly
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    # Complex rotation: (x1 + i*x2) * (cos + i*sin)
    # = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    out1 = x_part1 * cos - x_part2 * sin
    out2 = x_part1 * sin + x_part2 * cos

    return torch.cat([out1, out2], dim=-1)
