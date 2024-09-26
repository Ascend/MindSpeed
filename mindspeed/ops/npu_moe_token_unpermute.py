# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu

from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder

__all__ = ["npu_moe_token_unpermute"]


moe_token_unpermute_op_builder = MoeTokenUnpermuteOpBuilder()


def npu_moe_token_unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
):
    moe_token_unpermute_ops = moe_token_unpermute_op_builder.load()
    return moe_token_unpermute_ops.npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
