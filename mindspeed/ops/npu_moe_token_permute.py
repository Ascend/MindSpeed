# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu

from mindspeed.op_builder import MoeTokenPermuteOpBuilder

__all__ = ["npu_moe_token_permute"]


moe_token_permute_op_builder = MoeTokenPermuteOpBuilder()


def npu_moe_token_permute(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int = None,
        padded_mode: bool = False
):
    moe_token_permute_ops = moe_token_permute_op_builder.load()
    return moe_token_permute_ops.npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode)
