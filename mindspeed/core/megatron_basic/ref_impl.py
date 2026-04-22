# Copyright (c) 2023, Tri Dao.


import math
try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None

import torch
import torch.nn.functional as F


def hadamard_transform_ref(x, scale=1.0):
    """
    ref impl of fast_hadamard_transform_cuda
    x: (..., dim)
    out: (..., dim)
    """
    if hadamard is None:
        raise ImportError("Please install scipy")
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