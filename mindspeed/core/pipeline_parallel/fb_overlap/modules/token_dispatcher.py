# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from mindspeed.core.transformer.moe.unpermute_without_activation import UnpermuteWithoutActivation


def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
    # use 0.7.0 implement for better performance
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )

    ep_size = self.config.expert_model_parallel_size
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_extended_ep_size = ep_size * tp_size
    if self.drop_and_pad:
        self.capacity = self.probs.size(1)
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long,
            device=torch.cuda.current_device()
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        # Token drop but no pad. A synchronization is needed before the first
        # permutation to get the `num_out_tokens` CPU value.
        self.num_out_tokens = num_local_tokens_per_expert.sum().to(
            torch.device("cpu"), non_blocking=True
        )
        self.cuda_sync_point = "before_permutation_1"
    elif tp_extended_ep_size > 1:
        # Token dropless and enable ep. A synchronization is needed before expert parallel
        # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
        self.cuda_sync_point = "before_ep_alltoall"
    else:
        # Token dropless and no ep. A synchronization is needed before the token_permutation()
        # function returns to get the `tokens_per_expert` CPU value.
        self.cuda_sync_point = "before_finish"

    if tp_extended_ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(tp_extended_ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
            num_local_tokens_per_expert
        ).reshape(tp_extended_ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert
            .sum(axis=-1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================
    else:
        self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            -1, self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert

    if self.num_local_experts > 1:
        # No further synchronization is needed because torch.repeat_interleave() calls stream
        # synchronization internally when the `output_size` parameter is not provided.
        self.cuda_sync_point = "no_sync"
        self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
            self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
        )

    return num_tokens_per_local_expert


def alltoall_token_perm1(
    self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
):
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    tokens_per_expert = preprocess(self, indices)

    # Flatten the input tensor
    # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])


    # Permutation 1: input to AlltoAll input
    self.hiddden_shape_before_permute = hidden_states.shape
    if self.cuda_sync_point == "before_permutation_1":
        torch.cuda.current_stream().synchronize()
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states,
        indices,
        num_out_tokens=self.num_out_tokens,
        padded_mode=self.drop_and_pad,
    )

    # Perform expert parallel AlltoAll communication
    if self.cuda_sync_point == "before_ep_alltoall":
        torch.cuda.current_stream().synchronize()


    return permutated_local_input_tokens, tokens_per_expert


def alltoall_token_perm2(self, global_input_tokens):

    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
    if self.num_local_experts > 1:
        if not self.drop_and_pad:
            global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                global_input_tokens, self.global_input_tokens_local_experts_indices
            )
        else:
            global_input_tokens = global_input_tokens.reshape(
                self.ep_size, self.num_local_experts, self.capacity, -1
            )
            global_input_tokens = (
                global_input_tokens.transpose(0, 1)
                .reshape(self.num_local_experts * self.ep_size * self.capacity, -1)
                .contiguous()
            )

    if self.cuda_sync_point == "before_finish":
        torch.cuda.current_stream().synchronize()

    return global_input_tokens


def alltoall_token_unperm1(
    self,
    hidden_states: torch.Tensor,
    bias: torch.Tensor = None,
):
    """
    Reverse the token permutation to restore the original order.

    Args:
        hidden_states (torch.Tensor): Output from local experts.
        bias (torch.Tensor, optional): Bias tensor (not supported).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - Unpermuted token embeddings in the original order.
            - None (bias is not supported).
    """
    assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"


    # Unpermutation 2: expert output to AlltoAll input
    if self.num_local_experts > 1:
        if not self.drop_and_pad:
            hidden_states = unpermute(
                hidden_states,
                self.reversed_global_input_permutation_mapping,
            )
        else:
            hidden_states = hidden_states.reshape(
                self.num_local_experts, self.ep_size, self.capacity, -1
            )
            hidden_states = (
                hidden_states.transpose(0, 1)
                .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                .contiguous()
            )

    return hidden_states


def alltoall_token_unperm2(self, permutated_local_input_tokens, probs=None):
    # Unpermutation 1: AlltoAll output to output

    probs = probs if probs is not None else self.probs
    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        probs=probs,
        padded_mode=self.drop_and_pad,
        restore_shape=self.hiddden_shape_before_permute,
    )


    # Reshape the output tensor
    output = output.view(self.hidden_shape)

    output = make_viewless_tensor(
        inp=output, requires_grad=output.requires_grad, keep_graph=True
    )


    return output, None
