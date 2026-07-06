# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MindSpeed causal_conv1d — F.conv1d with per-sequence unpack for packed sequences.

There is no Triton-accelerated causal_conv1d in MindSpeed.  This module
provides a ``causal_conv1d`` function matching the FLA signature so that
MindSpeed's GatedDeltaNet subclass can import it directly.
"""

import torch
import torch.nn.functional as F


def _unpack_sequence(x, cu_seqlens, dim):
    """Split a packed tensor into per-sequence chunks along *dim*."""
    chunks = []
    for i in range(cu_seqlens.shape[0] - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(start, end)
        chunks.append(x[tuple(idx)])
    return chunks


def _conv1d_impl(x, weight_3d, bias):
    """Single (non-packed) causal conv1d via F.conv1d (fp32 internally)."""
    seq_len = x.shape[1]
    x_t = x.float().transpose(1, 2).contiguous()
    out = F.conv1d(
        input=x_t,
        weight=weight_3d.float(),
        bias=bias.float() if bias is not None else None,
        stride=1,
        padding=weight_3d.shape[-1] - 1,
        groups=weight_3d.shape[0],
    )
    out = out[..., :seq_len]
    return out.transpose(1, 2)


def causal_conv1d(x, weight, bias=None, activation=None, initial_state=None, output_final_state=False, cu_seqlens=None):
    """Torch-native causal_conv1d with FLA-compatible signature.

    Args:
        x: ``[b, s, d]`` (bshd layout).
        weight: ``[d, w]``.
        bias: optional ``[d]``.
        activation: ``"silu"`` / ``"swish"`` or ``None``.
        initial_state: ignored.
        output_final_state: always returns ``None``.
        cu_seqlens: ``[num_seqs + 1]`` or ``None``.

    Returns:
        ``(y, None)`` where *y* is ``[b, s, d]``.
    """
    weight_3d = weight.unsqueeze(1)

    if cu_seqlens is None:
        out = _conv1d_impl(x, weight_3d, bias)
    else:
        chunks = _unpack_sequence(x.float(), cu_seqlens, dim=1)
        out_chunks = [_conv1d_impl(c, weight_3d, bias) for c in chunks]
        out = torch.cat(out_chunks, dim=1)

    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")

    return out.to(x.dtype), None
