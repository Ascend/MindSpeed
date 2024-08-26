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
        topk: int = 1
) -> torch.Tensor:
        return npu_moe_token_unpermute(
                permuted_tokens, sorted_indices, probs, padded_mode=False, restore_shape=None)

    return wrapper
