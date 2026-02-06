# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch


def eager_permute(tokens, indices):
    topk = 1 if indices.dim() == 1 else indices.size(1)
    indices_dtype = indices.dtype
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True).to(indices_dtype)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def fused_permute(tokens, indices):
    raise NotImplementedError


def permute(tokens, indices, fused=False):
    return fused_permute(tokens, indices) if fused else eager_permute(tokens, indices)


def eager_unpermute(permuted_tokens, sorted_indices, probs):
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    unpermuted_tokens = torch.zeros([num_tokens, permuted_tokens.shape[-1]], dtype=permuted_tokens.dtype,
                                    device=permuted_tokens.device)
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens *= probs.unsqueeze(-1)
    return unpermuted_tokens.sum(dim=1)


def fused_unpermute(permuted_tokens, sorted_indices, probs):
    raise NotImplementedError


def unpermute(permuted_tokens, sorted_indices, probs=None, fused=False):
    if permuted_tokens.size(0) != sorted_indices.numel():
        raise AssertionError(f'permuted tokens({permuted_tokens.size(0)}) != sorted indices({sorted_indices.size()})')
    return fused_unpermute(permuted_tokens, sorted_indices, probs) if fused \
        else eager_unpermute(permuted_tokens, sorted_indices, probs)


def normalize_expert_args(top_k_index, top_k_weights):
    """
    Ensure top_k_index is integer tensor (indices) and top_k_weights is float tensor (weights).
    Swap if necessary and adjust dimensions if needed.

    Args:
        top_k_index: Tensor that could be either indices or weights
        top_k_weights: Tensor that could be either weights or indices

    Returns:
        (correct_top_k_index, correct_top_k_weights)
    """
    # Swap if top_k_index is floating point (actually weights)
    if torch.is_floating_point(top_k_index):
        top_k_index, top_k_weights = top_k_weights, top_k_index

    # Ensure weights have the same shape as indices
    if top_k_weights.size() != top_k_index.size():
        # Gather weights using indices (assume top_k_weights has shape [batch_size, num_experts])
        # and top_k_index has shape [batch_size, top_k]
        top_k_weights = top_k_weights.gather(1, top_k_index)

    return top_k_index, top_k_weights
