# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch


def permute(tokens, indices, topk: int = 1):
    if topk > 1:
        assert indices.size(1) == topk
    flatten_indices = indices.view(-1)
    # sorted_indices = torch.argsort(flatten_indices, stable=True)  # argsort int64 will be run on host cpu
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
    topk: int = 1
):
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    sorted_indices = torch.argsort(sorted_indices.float()).int()
    unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens