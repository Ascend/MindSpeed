# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu


__all__ = ["npu_moe_token_unpermute"]


def npu_moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    if (
        not padded_mode
        and probs is not None
        and probs.requires_grad
        and torch.is_grad_enabled()
        and permuted_tokens.shape[0] < sorted_indices.numel()
    ):
        # The inverse permutation identifies slots removed by token dropping.
        # WhereBackward clears their gradients, including uninitialized NaNs.
        retained = sorted_indices.reshape_as(probs) < permuted_tokens.shape[0]
        probs = torch.where(retained, probs, 0)
    return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
