# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies. All rights reserved.

from typing import List, Optional, Tuple

import torch
import torch.distributed
from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state,
    tensor_parallel,
    get_capacity,
    permute,
    unpermute,
    TransformerConfig,
)

from mindspeed.core.transformer.moe.comm_utils import AsyncAllToAllWithBackward


class MoEAlltoAllSEQMC2TokenDispatcher:
    """
    The legacy implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.

    Note: This class is a replica of the MoEAlltoAllTokenDispatcher from version 0.8.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        self.num_local_experts = num_local_experts
        self.config = config
        self.local_expert_indices = local_expert_indices
        super().__init__(num_local_experts, local_expert_indices, config)

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        assert self.config.moe_shared_expert_overlap
        self.shared_experts = shared_experts

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocess routing map for AlltoAll communication and token permutation.
        This method computes the number of tokens assigned to each expert based on
        the routing map. It also initializes the necessary data structures for
        AlltoAll communication, such as input and output splits, and the mapping
        between global tokens and local experts.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()
        # num_local_tokens_per_expert: [num_experts]
        send_splits = num_local_tokens_per_expert
        recv_splits = torch.empty_like(send_splits)
        torch.distributed.all_to_all_single(recv_splits, send_splits, group=self.ep_group)
        send_splits = send_splits.to('cpu', non_blocking=True)
        recv_splits = recv_splits.to('cpu', non_blocking=True)

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
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert_cpu = (
                self.num_global_tokens_per_local_expert.view(-1, self.num_local_experts).to(
                    torch.device("cpu"), non_blocking=True
                )
            )

        torch.cuda.current_stream().synchronize()
        self.send_counts = send_splits.tolist()
        self.recv_counts = recv_splits.tolist()

        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            routing_map (torch.Tensor): Mapping of tokens assigned to experts.
                Shape: [num_tokens, num_experts].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - Permuted probs of each token produced by the router.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for routing map"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        tokens_per_expert = self.preprocess(routing_map)

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Prepare share_expert comm.
        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()
        
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens, fused=self.config.moe_permute_fusion)

        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()

        _, global_probs, global_probs_async_handle = AsyncAllToAllWithBackward.apply(
            permuted_probs,
            self.output_splits,
            self.input_splits,
            parallel_state.get_expert_model_parallel_group(),
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(permutated_local_input_tokens)
        global_probs_async_handle.wait()

        # Permutation 2: Sort Probs by local expert.
        global_probs = torch.split(global_probs, self.num_global_tokens_per_local_expert_cpu.ravel().tolist(), dim=0)
        global_probs = torch.cat([global_probs[i] for i in self.sort_input_by_local_experts.tolist()], dim=0)

        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )

        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return permutated_local_input_tokens, tokens_per_expert, global_probs, self

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(
                hidden_states
            )

        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(hidden_states)
            self.shared_experts.post_forward_comm()

        # # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion
        )

        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output
        return output, None
