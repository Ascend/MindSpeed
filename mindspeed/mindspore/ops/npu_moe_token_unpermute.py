# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu

from mindspore import ops

__all__ = ["npu_moe_token_unpermute"]





def npu_moe_token_unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
):
    return ops.moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
