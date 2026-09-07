# Copyright (c) 2024, Huawei Technologies Co., Ltd.
#
# DSA RoPE optimizations.
#
# Provides the DSA indexer fused-RoPE path and an alternative complex-domain
# implementation.  The fused path is the Ascend equivalent of Megatron commit
# e354f1f57 (``perf(dsa): apply indexer RoPE in place``).  Megatron's change
# uses a Triton MLA kernel; MindSpeed routes the RoPE slice through the NPU
# fused rotary-position-embedding operator instead.
#
# Reference: MindSpeed 0.16 core_r0.16.0 dsa_matrix_naive.py

from functools import wraps

import torch


def dsa_transformer_config_post_init_wrapper(fn):
    """Allow DSA to retain ``apply_rope_fusion`` after Megatron validation.

    Megatron core_r0.18.0 rejects DSA and RoPE fusion before the DSA indexer is
    constructed.  MindSpeed supplies the missing NPU indexer implementation
    below, so hide only this flag while the native validation runs and restore
    it afterwards.  Other DSA validation, including the CP check, is preserved.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not (
            getattr(self, "experimental_attention_variant", None) == "dsa" and getattr(self, "apply_rope_fusion", False)
        ):
            return fn(self, *args, **kwargs)

        original_apply_rope_fusion = self.apply_rope_fusion
        self.apply_rope_fusion = False
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.apply_rope_fusion = original_apply_rope_fusion

    return wrapper


def _npu_fused_rope(x, cos, sin, mode):
    """Load the custom op lazily so CPU-only unit tests can import this module."""
    from mindspeed.ops.npu_rotary_position_embedding import (
        npu_rotary_position_embedding,
    )

    return npu_rotary_position_embedding(x, cos, sin, mode)


def apply_fused_rope(
    x,
    rotary_pos_emb,
    mscale=1.0,
    rotary_interleaved=False,
    multi_latent_attention=False,
):
    """Apply the Ascend fused RoPE operator to an already isolated RoPE slice."""
    if multi_latent_attention:
        # Match Megatron's MLA baseline layout conversion before the rotation.
        x = torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)
    cos = (torch.cos(rotary_pos_emb) * mscale).to(dtype=x.dtype)
    sin = (torch.sin(rotary_pos_emb) * mscale).to(dtype=x.dtype)
    mode = 1 if rotary_interleaved else 0
    return _npu_fused_rope(x.contiguous(), cos, sin, mode).to(dtype=x.dtype)


def dsa_indexer_apply_rope_wrapper(fn):
    """Add fused RoPE support to the core_r0.18.0 ``DSAIndexer`` layout.

    In core_r0.18.0 the indexer stores the non-RoPE dimensions first and the
    RoPE dimensions last.  Apply the NPU fused kernel only to the trailing RoPE
    slice, then restore the original indexer layout.
    """

    @wraps(fn)
    def wrapper(self, x, rotary_pos_emb, mscale, *args, **kwargs):
        if not getattr(self.config, "apply_rope_fusion", False):
            return fn(self, x, rotary_pos_emb, mscale, *args, **kwargs)

        no_pe_dim = self.index_head_dim - self.qk_pos_emb_head_dim
        x_nope, x_pe = torch.split(x, [no_pe_dim, self.qk_pos_emb_head_dim], dim=-1)
        x_pe = apply_fused_rope(
            x_pe,
            rotary_pos_emb,
            mscale=mscale,
            rotary_interleaved=getattr(self.config, "rotary_interleaved", False),
            multi_latent_attention=getattr(self.config, "multi_latent_attention", False),
        )
        return torch.cat([x_nope, x_pe], dim=-1)

    return wrapper


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
