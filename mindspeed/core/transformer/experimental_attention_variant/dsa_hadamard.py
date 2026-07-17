# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.nn.functional as F


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Torch fallback for DSA Hadamard rotation.

    Megatron's DSA code owns the rotation semantics and calls this backend through
    its module-level ``hadamard_transform`` variable.
    """
    original_shape = x.shape
    hidden_size = original_shape[-1]
    dim_padded = 1 << (hidden_size - 1).bit_length()

    if hidden_size != dim_padded:
        x = F.pad(x, (0, dim_padded - hidden_size))

    dtype = x.dtype
    x = x.reshape(-1, dim_padded).float()

    stride = 1
    while stride < dim_padded:
        x = x.reshape(-1, dim_padded // (stride * 2), stride * 2)
        left = x[..., :stride]
        right = x[..., stride:]
        x = torch.cat((left + right, left - right), dim=-1)
        stride *= 2

    x = x.reshape(*original_shape[:-1], dim_padded)[..., :hidden_size]
    return (x * scale).to(dtype)
