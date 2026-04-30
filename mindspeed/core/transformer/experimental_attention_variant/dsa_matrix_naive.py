# Copyright (c) 2023; Tri Dao. All rights reserved.
# Copyright (c) 2024; Huawei Technologies Co., Ltd.  All rights reserved.

import math

import torch
import torch.nn.functional as F

from scipy.linalg import hadamard

from mindspeed.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_bshd_in_complex


def hadamard_transform_ref(x, scale=1.0):
    """
    Eager implementation of the Hadamard transform

    Args:
        x:(torch.Tensor): input tensor
    """

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale

    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a scaled Hadamard transform to the input tensor, commonly used for rotating activations

    Args:
        x (torch.Tensor): Input tensor of shape [..., hidden_size], must be of dtype torch.bfloat16.
    """

    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        hadamard_transform = hadamard_transform_ref

    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


def apply_rope_in_complex(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float) -> torch.Tensor:
    """Apply RoPE to the input tensor in complex"""
    x_pe, x_nope = torch.split(
        x, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
    )
    x_pe = apply_rotary_pos_emb_bshd_in_complex(x_pe, rotary_pos_emb, rotary_interleaved=True)
    x = torch.cat([x_pe, x_nope], dim=-1)
    return x
