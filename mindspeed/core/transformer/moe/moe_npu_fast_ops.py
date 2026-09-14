# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""NPU fast-path replacements for Megatron 0.15+ MoE op selections.

Megatron 0.15+ builds the router ``routing_probs`` / ``routing_map`` with
``index_put_`` and the MoE ``unpermute`` with ``index_add_`` when
``torch.use_deterministic_algorithms(True)`` is set, and Megatron 0.17+
replaced the ``masked_select`` based ``permute`` with an ``argsort`` based
implementation for CUDA graph compatibility. On NPU those selections lower to
slow AICPU kernels (``aclnnSort`` / ``aclnnIndexPutImpl``), while
``masked_select`` / ``Tensor.scatter`` / ``scatter_add_`` keep the fast
vector-core paths and provide the same determinism level that MindSpeed
master (mcore 0.12.x) runs with under ``--npu-deterministic``.
"""

from functools import wraps
from typing import Optional, Tuple

import torch

from megatron.core.transformer.moe.moe_utils import permute as megatron_permute
from megatron.core.transformer.moe.moe_utils import unpermute as megatron_unpermute


def permute_npu(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    tokens_per_expert: Optional[torch.Tensor] = None,
    align_size: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Permute tokens grouped by expert, restoring the Megatron 0.12 ``masked_select`` fast path.

    Keeps the Megatron 0.17+ five-value return contract. The dropless branch uses
    ``masked_select`` instead of ``argsort`` to avoid the AICPU ``aclnnSort`` kernel
    on NPU (CUDA graphs, the motivation for the upstream change, are not used in NPU
    training). The ``drop_and_pad`` and ``fused`` branches behave exactly like
    Megatron 0.18.
    """
    if fused:
        return megatron_permute(
            tokens,
            routing_map,
            probs=probs,
            num_out_tokens=num_out_tokens,
            fused=fused,
            drop_and_pad=drop_and_pad,
            tokens_per_expert=tokens_per_expert,
            align_size=align_size,
        )

    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)
        if probs is not None:
            routing_map = routing_map.bool()
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()
        token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map)
    else:
        permuted_probs = None

    return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert


def unpermute_npu(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    pad_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Restore the original token order, always using ``scatter_add_``.

    Megatron 0.15+ selects ``index_add_`` under deterministic algorithms, which lowers
    to a slower NPU kernel than ``aclnnScatterAdd``. Behavior otherwise matches
    Megatron 0.18 (and the op selection matches MindSpeed master / mcore 0.12).
    """
    if fused:
        return megatron_unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=probs,
            routing_map=routing_map,
            fused=fused,
            drop_and_pad=drop_and_pad,
            pad_offsets=pad_offsets,
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros and scatter add the permuted
    # input back to the original positions.
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


def topk_routing_with_score_function_scatter_wrapper(fn):
    """Build the router routing tensors with ``Tensor.scatter`` instead of deterministic ``index_put_``.

    Megatron 0.15+ constructs ``routing_probs`` / ``routing_map`` via ``index_put_``
    when deterministic algorithms are enabled, which lowers to the AICPU
    ``aclnnIndexPutImpl`` kernel on NPU. The wrapped function is called with
    ``dense_output=True`` (bypassing that branch) and the sparse tensors are rebuilt
    with ``scatter``, matching the Megatron 0.12 op selection.
    """

    @wraps(fn)
    def wrapper(
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool = False,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        scaling_factor: Optional[float] = None,
        score_function: str = "softmax",
        expert_bias: Optional[torch.Tensor] = None,
        fused: bool = False,
        router_replay: Optional[object] = None,
        dense_output: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fused:
            return fn(
                logits,
                topk,
                use_pre_softmax=use_pre_softmax,
                num_groups=num_groups,
                group_topk=group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
                expert_bias=expert_bias,
                fused=fused,
                router_replay=router_replay,
                dense_output=dense_output,
                **kwargs,
            )

        probs, top_indices = fn(
            logits,
            topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
            fused=fused,
            router_replay=router_replay,
            dense_output=True,
            **kwargs,
        )
        if dense_output:
            return probs, top_indices

        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        return routing_probs, routing_map

    return wrapper
