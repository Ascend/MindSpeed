# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half
from mindspeed.args_utils import get_full_args
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
from mindspeed.core.transformer.multi_head_latent_attention.mla_utils import yarn_get_mscale


def fused_apply_rotary_pos_emb_wrapper(fn):
    """Use the native NPU adjoint for ordinary SBHD RoPE.

    Megatron has already selected fusion and sliced CP frequencies before
    this entry point. The native autograd op saves forward cos/sin and calls
    RotaryPositionEmbeddingGrad directly; it does not inspect device values
    in Python or recompute the trigonometric tensors during backward.
    Other layouts retain the Transformer Engine implementation.
    """

    @wraps(fn)
    def wrapper(t, freqs, transpose_output_memory=False, interleaved=False):
        # Keep the compatibility guard short-circuiting without extra hot-path calls.
        if (
            transpose_output_memory  # pylint: disable=too-many-boolean-expressions
            or t.ndim != 4
            or freqs.ndim != 4
            or freqs.shape[1:3] != (1, 1)
            or freqs.shape[0] < t.shape[0]
            or not 0 < freqs.shape[-1] <= t.shape[-1]
            or freqs.shape[-1] % 2
            or freqs.requires_grad
            or t.numel() == 0
            or t.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return fn(t, freqs, transpose_output_memory, interleaved)

        # Match TE's float32 frequency computation and partial-RoPE contract.
        freqs = freqs[: t.shape[0]].float()
        cos = freqs.cos().to(t.dtype)
        sin = freqs.sin().to(t.dtype)
        rot_dim = freqs.shape[-1]
        rotated = npu_rotary_position_embedding(t[..., :rot_dim].contiguous(), cos, sin, 1 if interleaved else 0)
        if rot_dim == t.shape[-1]:
            return rotated
        return torch.cat((rotated, t[..., rot_dim:]), dim=-1)

    return wrapper


def apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]
        rotary_interleaved (bool, optional): A flag indicating whether to interleave the
            application of sine and cosine components. Defaults to False.
        multi_latent_attention (bool, optional): A flag indicating if multi-latent attention
            mechanism should be applied. If True, it splits the tensor into multiple latent
            spaces before applying RoPE. Defaults to False.
        mscale (float, optional): A scaling factor applied to both the sine and cosine
            components of the positional embeddings. Defaults to 1.0.
    Returns:
        Tensor: The input tensor after applying RoPE
    """
    args = get_full_args()
    _mscale = mscale
    if hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "yarn":
        _mscale = float(
            yarn_get_mscale(args.yarn_scaling_factor, args.rope_scaling_mscale)
            / yarn_get_mscale(args.yarn_scaling_factor, args.rope_scaling_mscale_all_dim)
        )

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = (torch.cos(freqs) * _mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)

    if getattr(args, "apply_rope_fusion", False):
        mode = 1 if rotary_interleaved else 0
        t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
    else:
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    return torch.cat((t, t_pass), dim=-1)
