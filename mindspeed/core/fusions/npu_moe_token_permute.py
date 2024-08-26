# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps

import torch
import torch_npu

from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute


def permute_wrapper(fn):
    @wraps(fn)
    def wrapper(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        topk: int = 1
    ) -> torch.Tensor:
        return npu_moe_token_permute(tokens, indices)

    return wrapper
