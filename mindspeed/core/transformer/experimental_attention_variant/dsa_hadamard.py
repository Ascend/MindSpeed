# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def _hadamard_matrix(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the cached ``dim x dim`` Hadamard (+-1) matrix; ``dim`` must be 2^k.

    Built once per (dim, dtype, device) by recursive doubling, so the matrix
    is not rebuilt or copied host-to-device on every call.
    """
    mat = torch.ones(1, 1, dtype=dtype, device=device)
    while mat.shape[0] < dim:
        mat = torch.cat((torch.cat((mat, mat), dim=1), torch.cat((mat, -mat), dim=1)), dim=0)
    return mat


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Torch fallback for DSA Hadamard rotation.

    Megatron's DSA code owns the rotation semantics and calls this backend through
    its module-level ``hadamard_transform`` variable.

    Implemented as a single GEMM against the cached Hadamard matrix, keeping
    the forward and the autograd backward cheap compared with an eager
    log-depth butterfly loop.
    """
    original_shape = x.shape
    hidden_size = original_shape[-1]
    dim_padded = 1 << (hidden_size - 1).bit_length()

    if hidden_size != dim_padded:
        x = F.pad(x, (0, dim_padded - hidden_size))

    mat = _hadamard_matrix(dim_padded, x.dtype, x.device)
    out = F.linear(x.reshape(-1, dim_padded), mat) * scale
    return out[..., :hidden_size].reshape(original_shape)
