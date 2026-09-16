# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Torch-native causal-convolution fallback for GatedDeltaNet."""

import torch
import torch.nn.functional as F


def _unpack_sequence(x, cu_seqlens, dim):
    """Split a packed tensor into its original sequences along ``dim``."""
    chunks = []
    for i in range(cu_seqlens.shape[0] - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        index = [slice(None)] * x.dim()
        index[dim] = slice(start, end)
        chunks.append(x[tuple(index)])
    return chunks


def _conv1d_impl(x, weight_3d, bias):
    """Apply one causal depthwise convolution in fp32 internally."""
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
    return out[..., :seq_len].transpose(1, 2)


def causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """Run GDN's causal convolution for fixed or packed BSHD inputs."""
    if initial_state is not None or output_final_state:
        raise NotImplementedError("GDN causal_conv1d does not support convolution state")

    weight_3d = weight.unsqueeze(1)
    if cu_seqlens is None:
        out = _conv1d_impl(x, weight_3d, bias)
    else:
        if x.shape[0] != 1:
            raise ValueError("Packed causal_conv1d expects batch dimension 1")
        chunks = _unpack_sequence(x, cu_seqlens, dim=1)
        out = torch.cat([_conv1d_impl(chunk, weight_3d, bias) for chunk in chunks], dim=1)

    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")

    return out.to(x.dtype), None
