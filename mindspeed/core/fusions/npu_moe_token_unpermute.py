# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps

import torch
from megatron.training import get_args

from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute


def unpermute_wrapper(fn):
    @wraps(fn)
    def wrapper(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        topk: int = 1
) -> torch.Tensor:
        _args = get_args()
        if _args.use_fused_moe_token_permute_and_unpermute:
            return npu_moe_token_unpermute(
                permuted_tokens, sorted_indices, probs, padded_mode=False, restore_shape=None)
        return fn(permuted_tokens, sorted_indices, probs, topk=topk)

    return wrapper
