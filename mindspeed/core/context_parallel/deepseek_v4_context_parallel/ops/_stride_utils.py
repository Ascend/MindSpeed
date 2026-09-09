# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Tensor stride helpers for DeepSeek-V4 CANN operator boundaries."""

from functools import lru_cache

import torch


@lru_cache(maxsize=128)
def _canonical_dense_strides(shape):
    strides = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = running
        running *= int(shape[index])
    return tuple(strides)


def normalize_dense_strides(tensor):
    """Return a tensor with canonical row-major strides.

    PyTorch treats arbitrary strides on size-one dimensions as contiguous, but
    some CANN tiling checks validate the raw strides. The 2-D round-trip repairs
    that case without copying and materializes only genuinely non-dense inputs.
    """
    if tensor is None or tensor.dim() == 0 or tensor.numel() == 0:
        return tensor

    shape = tensor.shape
    expected_strides = _canonical_dense_strides(shape)
    if tensor.stride() == expected_strides:
        return tensor

    if tensor.is_contiguous():
        return tensor.as_strided(shape, expected_strides)

    return tensor.clone(memory_format=torch.contiguous_format)
