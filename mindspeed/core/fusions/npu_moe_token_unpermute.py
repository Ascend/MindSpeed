# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps

import torch

from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute


def unpermute_wrapper(fn):
    @wraps(fn)
    def wrapper(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
) -> torch.Tensor:
        dtype = permuted_tokens.dtype
        if probs is not None and probs.dtype != permuted_tokens.dtype:
            # make sure permuted_tokens has the same dtype with probs.
            permuted_tokens = permuted_tokens.to(probs.dtype)
        return npu_moe_token_unpermute(
                permuted_tokens, sorted_indices, probs, padded_mode=padded_mode, restore_shape=restore_shape).to(dtype)

    return wrapper
