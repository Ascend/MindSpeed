# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies.
# All rights reserved.
import torch

from mindspeed_llm.mindspore.core.tensor_parallel.mappings import all_to_all

from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state,
    tensor_parallel,
    permute,
    unpermute,
    sort_chunks_by_idxs,
    get_capacity)

from mindspeed.core.transformer.moe.moe_feature import MoEAlltoAllSEQTokenDispatcher as MegatronMoEAlltoAllSEQTokenDispatcher


class All2AllSeqTp2epDispatcherImpl:
    """
    original logic is alltoall in tp region, then alltoallv in ep region
    if use tp_extend_ep, just alltoallv in tp*ep region
    """

    def __init__(self, num_local_experts, local_expert_indices, config):
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

        # use MOEAlltoAllSEQTokenDispatcher to init
        self.drop_and_pad = None
        self.num_experts = None
        self.capacity = None
        self.tp_size = None
        self.ep_size = None
        self.num_out_tokens = None
        self.input_splits = None
        self.output_splits = None
        self.cuda_sync_point = None
        self.num_global_tokens_per_local_expert = None
        self.num_global_tokens_per_local_expert_cpu = None
        self.hidden_shape = None
        self.probs = None
        super().__init__(num_local_experts, local_expert_indices, config)

    def preprocess(self, routing_map):
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
        ep_size = self.config.expert_model_parallel_size
        if self.drop_and_pad:
            # probs: [num_experts, capacity]
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
            # Token drop but no pad.
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
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_extended_ep_size = ep_size * tp_size
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
            num_global_tokens_per_expert = tensor_parallel.gather_from_sequence_parallel_region(
                num_local_tokens_per_expert, group=parallel_state.get_expert_tensor_and_model_parallel_group()
            ).reshape(tp_extended_ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                                                      :, self.local_expert_indices
                                                      ]
            
            group_size = tp_extended_ep_size  # = ep_size * tp_size
            experts_per_rank = self.num_experts // group_size  # uniformly distributed experts for each rank

            rank2rank = num_global_tokens_per_expert \
                .reshape(group_size, group_size, experts_per_rank) \
                .sum(dim=-1)  # [group_size, group_size]

            self.send_count_matrix = rank2rank.reshape(-1).tolist()       # row-major order
            self.send_count_matrix_T = rank2rank.T.reshape(-1).tolist()     # Transposited send_count_matrix
  
            del rank2rank, experts_per_rank, group_size
            
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
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

            if not hasattr(self, 'comm_stream'):
                self.comm_stream = torch.cuda.Stream()
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                expert_ids_per_ep_rank = torch.tensor(
                    [i % self.num_local_experts for i in range(self.config.num_moe_experts)],
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
                self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                    expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
                )

        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor,
    ):
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            routing_map (torch.Tensor): Mapping of tokens assigned to experts.
                Shape: [num_tokens, num_experts].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(routing_map)

        # Flatten the input tensor
        # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens, fused=self.config.moe_permute_fusion)


        # Perform expert parallel AlltoAll communication
        global_input_tokens = all_to_all(
            parallel_state.get_expert_tensor_and_model_parallel_group(),
            permutated_local_input_tokens,
            self.send_count_matrix,
        )
        global_probs = all_to_all(
            parallel_state.get_expert_tensor_and_model_parallel_group(),
            permuted_probs,
            self.send_count_matrix,
        )

        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if self.num_local_experts > 1:
            global_input_tokens, global_probs = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
                probs=global_probs,
            )

        return global_input_tokens, tokens_per_expert, global_probs

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
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
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if self.num_local_experts > 1:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )

        # Perform expert parallel AlltoAll communication
        permutated_local_input_tokens = all_to_all(
            parallel_state.get_expert_tensor_and_model_parallel_group(),
            hidden_states,
            self.send_count_matrix_T,
        )

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None


# Define the substitute wrapper class (in same file or a separate file)
# The original class MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher is located in 'MindSpeed/mindspeed/core/transformer/moe/moe_feature/adaptor.py'


class _PatchedMOEAlltoAllSEQTptoEpTokenDispatcher(All2AllSeqTp2epDispatcherImpl,
                                                  MegatronMoEAlltoAllSEQTokenDispatcher):
    def __init__(self, *args, **kwargs):
        # 保险起见：两边都显式调一遍，避免对方不是 cooperative super
        MegatronMoEAlltoAllSEQTokenDispatcher.__init__(self, *args, **kwargs)
        All2AllSeqTp2epDispatcherImpl.__init__(self, *args, **kwargs)