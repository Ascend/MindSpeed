# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu

from mindspore import ops

__all__ = ["npu_moe_token_permute"]





def npu_moe_token_permute(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int = None,
        padded_mode: bool = False
):
    num_out_tokens = 0 if num_out_tokens is None else num_out_tokens
    return ops.moe_token_permute(tokens, indices, num_out_tokens, padded_mode)