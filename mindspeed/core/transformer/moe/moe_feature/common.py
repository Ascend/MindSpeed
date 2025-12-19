# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.core.transformer.moe.moe_feature import (
    gather_from_sequence_parallel_region,
    _split_along_first_dim,
    tensor_parallel,
    get_capacity
)


def routing_tp_extend_ep(self, logits: torch.Tensor):
    """
    if use tp_extend_ep, logits is not need to gather from the tp region
    """
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)
    logits = gather_from_sequence_parallel_region(logits)

    if self.routing_type == "sinkhorn":
        scores, routing_map = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, routing_map = self.aux_loss_load_balancing(logits)
    elif self.routing_type == "none":
        # A naive top-k routing without load balancing
        scores, routing_map, _ = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

    scores = _split_along_first_dim(scores)
    routing_map = _split_along_first_dim(routing_map)
    return scores, routing_map


def ascend_gmm_preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
    """
    Preprocess routing map for AlltoAll communication and token permutation.
    This method computes the number of tokens assigned to each expert based on
    the routing map. It also initializes the necessary data structures for
    AlltoAll communication, such as input and output splits, and the mapping
    between global tokens and local experts.

    In ascend GMM, num_tokens_per_local_expert always in NPU.

    Args:
        routing_map (torch.Tensor): The mapping of tokens to experts, with shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: Tensor containing the number of tokens assigned to local expert.
    """
    num_local_tokens_per_expert = routing_map.sum(dim=0).long()
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    if self.drop_and_pad:
        # Drop and pad the input to capacity.
        num_tokens = routing_map.size(0) * self.config.moe_router_topk
        self.capacity = get_capacity(
            num_tokens=num_tokens,
            num_experts=self.num_experts,
            capacity_factor=self.config.moe_expert_capacity_factor,
        )
        self.num_out_tokens = self.capacity * self.num_experts
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
        )
        self.num_global_tokens_per_local_expert_cpu = torch.full(
            (self.num_experts * self.tp_size,), self.capacity, dtype=torch.long
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        # Token drop but no pad. A synchronization is needed before the first
        # permutation to get the `num_out_tokens` CPU value.
        self.num_out_tokens = num_local_tokens_per_expert.sum().to(
            torch.device("cpu"), non_blocking=True
        )
        self.cuda_sync_point = "before_permutation_1"
    else:
        # Dropless
        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk
        if self.ep_size > 1 or self.num_local_experts > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.cuda_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed to get the
            # `tokens_per_expert` CPU value.
            self.cuda_sync_point = "before_finish"

    if ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = tensor_parallel.gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1)
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
        self.num_global_tokens_per_local_expert_cpu = (
            self.num_global_tokens_per_local_expert.view(-1, self.num_local_experts).to(
                torch.device("cpu"), non_blocking=True
            )
        )

    return num_tokens_per_local_expert
