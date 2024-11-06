# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps

import torch

from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute


def permute_wrapper(fn):
    @wraps(fn)
    def wrapper(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int = None,
        padded_mode: bool = False
    ) -> torch.Tensor:
        return npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode)

    return wrapper