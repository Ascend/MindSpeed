# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state,
    gather_from_sequence_parallel_region,
    tensor_parallel,
    permute,
    unpermute,
    sort_chunks_by_idxs,
    MoEAlltoAllTokenDispatcher,
    get_capacity)
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather, async_reduce_scatter


PREMUTE_FINISH_EVENT = None


class MindSpeedMOEAlltoAllFbOverlapTokenDispatcher(MoEAlltoAllTokenDispatcher):
    OVERLAP_STREAM = None

    def __init__(self, num_local_experts, local_expert_indices, config):
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super(MindSpeedMOEAlltoAllFbOverlapTokenDispatcher, self).__init__(num_local_experts, local_expert_indices, config)
        if MindSpeedMOEAlltoAllFbOverlapTokenDispatcher.OVERLAP_STREAM is None:
            MindSpeedMOEAlltoAllFbOverlapTokenDispatcher.OVERLAP_STREAM = torch.npu.Stream(device=torch.npu.current_device())
        self.overlap_stream = MindSpeedMOEAlltoAllFbOverlapTokenDispatcher.OVERLAP_STREAM

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
        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts], number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            if self.config.moe_pad_expert_input_to_capacity:
                num_tokens_per_local_expert = num_tokens_per_local_expert.to('npu')
            if self.num_local_experts > 1:
                if self.num_global_tokens_per_local_expert.device.type != 'cpu':
                    self.num_global_tokens_per_local_expert = (
                        self.num_global_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True))
                self.num_global_tokens_per_local_expert_cpu = self.num_global_tokens_per_local_expert
            return num_tokens_per_local_expert

        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if self.config.moe_expert_capacity_factor is not None:
            # Drop tokens to capacity, no padding.
            self.num_out_tokens = num_local_tokens_per_expert.sum()

            # A synchronization is needed before the first permutation
            # to get the `num_out_tokens` CPU value.
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # Dropless
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        if self.ep_size > 1 or self.num_local_experts > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.cuda_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed before the returns
            # to get the `tokens_per_expert` CPU value for
            self.cuda_sync_point = "before_finish"

        if self.ep_size > 1 or self.tp_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.tp_ep_group
                )
                .reshape(self.ep_size, self.tp_size, self.num_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                                                :, :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
                                                ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            self.num_tokens_per_expert = num_global_tokens_per_expert.reshape(-1, self.num_experts).sum(axis=0).clone()

            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            self.output_splits = (
                num_global_tokens_per_rank[self.tp_rank]
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = (
                num_global_tokens_per_rank.sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # Megatron will copy num_tokens_per_local_expert to cpu for GMM.
            # NPU GMM can support input splits tensor in device, so no need to copy.
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not self.config.moe_permute_fusion:
                self.num_global_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True)
            self.num_global_tokens_per_local_expert_cpu = self.num_global_tokens_per_local_expert

        return num_tokens_per_local_expert



    def token_permute1(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor,
    ):
        """
        Dispatch tokens to local experts before alltoall comm

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
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()

        event = torch.npu.current_stream().record_event()

        with torch.npu.stream(self.overlap_stream):
            global PREMUTE_FINISH_EVENT
            self.overlap_stream.wait_event(event)
            if self.config.moe_permute_fusion:
                permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute(
                    hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens,
                    fused=self.config.moe_permute_fusion, drop_and_pad=self.drop_and_pad,
                )
            else:
                permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute(
                    hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens,
                    drop_and_pad=self.drop_and_pad,
                )
            PREMUTE_FINISH_EVENT = self.overlap_stream.record_event()

        return permutated_local_input_tokens, permuted_probs, tokens_per_expert

    def async_dispatch_comm(
        self, permutated_local_input_tokens, permutated_local_input_token_probs=None,
        input_splits=None, output_splits=None, output_splits_tp=None, wait_event=None
    ):
        input_splits = input_splits if input_splits is not None else self.input_splits
        output_splits = output_splits if output_splits is not None else self.output_splits
        output_splits_tp = output_splits_tp if output_splits_tp is not None else self.output_splits_tp
        _, global_input_tokens, comm_handle = async_all_to_all(
            permutated_local_input_tokens,
            output_splits,
            input_splits,
            self.ep_group,
            event=wait_event,
            stream=torch.npu.current_stream() if wait_event else None
        )

        global_input_token_probs, prob_comm_handle = None, None
        if permutated_local_input_token_probs is not None:
            _, global_input_token_probs, prob_comm_handle = async_all_to_all(
                permutated_local_input_token_probs,
                output_splits,
                input_splits,
                self.ep_group,
                event=comm_handle,
                stream=torch.npu.current_stream()
            )

        if self.tp_size > 1:
            _, global_input_tokens, comm_handle = async_all_gather(
                global_input_tokens, self.tp_group, output_split_sizes=output_splits_tp.tolist() if output_splits_tp is not None else None,
                event=prob_comm_handle if prob_comm_handle else comm_handle, stream=torch.npu.current_stream()
            )
            if global_input_token_probs is not None:
                _, global_input_token_probs, prob_comm_handle = async_all_gather(
                    global_input_token_probs, self.tp_group, output_split_sizes=output_splits_tp.tolist() if output_splits_tp is not None else None,
                    event=prob_comm_handle, stream=torch.npu.current_stream()
                )

        return (global_input_tokens, comm_handle), (global_input_token_probs, prob_comm_handle)

    def backward_async_dispatch_comm(
        self, tokens_grad, token_probs_grad=None,
        input_splits=None, output_splits=None, input_splits_tp=None, wait_event=None
    ):
        last_comm_handle = wait_event
        if self.tp_size > 1:
            input_split_sizes = input_splits_tp.tolist() if input_splits_tp is not None else None
            _, tokens_grad, last_comm_handle = async_reduce_scatter(
                tokens_grad, self.tp_group, input_split_sizes=input_split_sizes,
                event=wait_event, stream=torch.npu.current_stream() if wait_event else None
            )
            if token_probs_grad is not None:
                _, token_probs_grad, last_comm_handle = async_reduce_scatter(
                    token_probs_grad, self.tp_group, input_split_sizes=input_split_sizes,
                    event=wait_event, stream=torch.npu.current_stream() if wait_event else None
                )

        _, tokens_grad, tokens_comm_handle = async_all_to_all(
            tokens_grad, output_splits, input_splits, self.ep_group, event=last_comm_handle,
            stream=torch.npu.current_stream() if last_comm_handle else None
        )
        token_probs_comm_handle = None
        if token_probs_grad is not None:
            _, token_probs_grad, token_probs_comm_handle = async_all_to_all(
                token_probs_grad, output_splits, input_splits, self.ep_group, event=last_comm_handle,
                stream=torch.npu.current_stream() if last_comm_handle else None
            )

        return (tokens_grad, tokens_comm_handle), (token_probs_grad, token_probs_comm_handle)


    def token_permute2(self, global_input_tokens, global_input_token_probs, num_global_tokens_per_local_expert=None):
        num_global_tokens_per_local_expert = num_global_tokens_per_local_expert \
            if num_global_tokens_per_local_expert is not None else self.num_global_tokens_per_local_expert

        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
                global_input_token_probs = (
                    global_input_token_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_token_probs.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                ) if global_input_token_probs is not None else None
            else:
                global_input_tokens, global_input_token_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_input_token_probs,
                    fused=self.config.moe_permute_fusion,
                )

        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens, global_input_token_probs


    def token_unpermute1(
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
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states, _ = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        return hidden_states

    def async_combine_comm(self, hidden_states, input_splits=None, output_splits=None, input_splits_tp=None, wait_event=None):

        # Perform expert parallel AlltoAll communication
        output_splits = output_splits if output_splits is not None else self.input_splits
        input_splits = input_splits if input_splits is not None else self.output_splits
        input_splits_tp = input_splits_tp if input_splits_tp is not None else self.output_splits_tp


        if self.tp_size > 1:
            _, hidden_states, wait_event = async_reduce_scatter(
                hidden_states, self.tp_group, input_split_sizes=input_splits_tp.tolist() if input_splits_tp is not None else None,
                event=wait_event, stream=torch.npu.current_stream() if wait_event else None
            )

        _, permutated_local_input_tokens, comm_handle = async_all_to_all(
            hidden_states,
            output_splits,
            input_splits,
            self.ep_group,
            event=wait_event,
            stream=torch.npu.current_stream() if wait_event else None
        )

        return permutated_local_input_tokens, comm_handle

    def backward_async_combine_comm(self, tokens_grad, input_splits=None, output_splits=None, output_splits_tp=None, wait_event=None):

        _, tokens_grad, comm_handle = async_all_to_all(
            tokens_grad, output_splits, input_splits, self.ep_group, event=wait_event,
            stream=torch.npu.current_stream() if wait_event else None
        )

        if self.tp_size > 1:
            output_split_sizes = output_splits_tp.tolist() if output_splits_tp is not None else None
            _, tokens_grad, comm_handle = async_all_gather(
                tokens_grad, self.tp_group, output_split_sizes=output_split_sizes,
                event=comm_handle, stream=torch.npu.current_stream()
            )

        return tokens_grad, comm_handle

    def token_unpermute2(self, permutated_local_input_tokens):

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None