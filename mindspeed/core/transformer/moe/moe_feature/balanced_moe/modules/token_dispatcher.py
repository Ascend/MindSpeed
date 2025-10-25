# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import List, Tuple

import numpy as np
import torch

from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather, async_reduce_scatter
from mindspeed.core.transformer.moe.moe_feature import (
    TransformerConfig,
    parallel_state,
    gather_from_sequence_parallel_region,
    tensor_parallel,
    permute,
    unpermute,
    sort_chunks_by_idxs,
    MoEAlltoAllTokenDispatcher)
from mindspeed.core.transformer.moe.moe_feature.balanced_moe.utils import CustomSliceFunction
from mindspeed.core.transformer.moe.moe_feature.overlap.token_dispatcher import is_less_or_equal_rc2_cann_version

cann_version_check = is_less_or_equal_rc2_cann_version()
PREMUTE_FINISH_EVENT = None


def move_sorted_columns_to_end(routing_map: torch.Tensor,
                               remote_list: torch.Tensor) -> torch.Tensor:
    num_experts = routing_map.size(1)
    device = routing_map.device

    all_indices = torch.arange(num_experts, device=device)

    mask = torch.ones(num_experts, dtype=torch.bool, device=device)
    mask[remote_list] = False

    new_indices = torch.cat([
        all_indices[mask],
        all_indices[~mask]
    ])

    return routing_map[:, new_indices]


class MoEBalancedAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    OVERLAP_STREAM = None

    def __init__(self, num_local_experts: int, local_expert_indices: List[int],
                 config: TransformerConfig):
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """

        super(MoEBalancedAlltoAllTokenDispatcher, self).__init__(num_local_experts, local_expert_indices, config)
        if MoEBalancedAlltoAllTokenDispatcher.OVERLAP_STREAM is None:
            MoEBalancedAlltoAllTokenDispatcher.OVERLAP_STREAM = torch.npu.Stream(
                device=torch.npu.current_device())
        self.overlap_stream = MoEBalancedAlltoAllTokenDispatcher.OVERLAP_STREAM

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
        self.num_local_tokens_per_expert = num_local_tokens_per_expert
        self.num_local_tokens_per_expert_np = (
            num_local_tokens_per_expert
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
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

        hot_expert_num = self.config.balanced_moe_hot_expert_num
        if self.ep_size > 1 or self.tp_size > 1:
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

            # cpu/npu syncronize here
            num_global_tokens_per_expert_cpu = num_global_tokens_per_expert.to(
                torch.device("cpu")).numpy()  # [tp_extended_ep_size, num_experts]
            self.hot_experts = self.hot_expert_selection(num_global_tokens_per_expert_cpu[self.tp_rank], hot_expert_num)
            if isinstance(self.local_expert_indices, list):
                self.local_expert_indices_np = np.array(self.local_expert_indices)
            else:
                self.local_expert_indices_np = self.local_expert_indices.numpy()
            self.cold_experts = self.local_expert_indices_np[~np.isin(self.local_expert_indices_np, self.hot_experts)]
            self.num_cold_experts = self.cold_experts.shape[0]
            local_hot_mask = np.isin(self.hot_experts, self.local_expert_indices_np)
            self.local_hot_experts = self.hot_experts[local_hot_mask]
            self.remote_hot_experts = self.hot_experts[~local_hot_mask]

            self.sorted_cold_hot_experts = np.concatenate((self.hot_experts, self.cold_experts)).sort()
            local_from_hot_mask = np.isin(self.hot_experts, self.local_hot_experts)
            self.padded_num_remote_hot_tokens = self.num_local_tokens_per_expert_np[self.hot_experts].copy()
            self.padded_num_remote_hot_tokens[local_from_hot_mask] = 0
            remote_hot_masked_num_local_tokens_per_expert = self.num_local_tokens_per_expert_np.copy()  # [num_experts]
            remote_hot_masked_num_local_tokens_per_expert[self.remote_hot_experts] = 0

            self.input_splits = (
                remote_hot_masked_num_local_tokens_per_expert
                .reshape(self.ep_size, self.num_local_experts)
                .sum(axis=1)
            )

            curr_ep_rank = parallel_state.get_expert_model_parallel_rank()
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]

            num_global_cold_local_hot_tokens_per_local_expert = num_global_tokens_per_expert[
                                                                :, :, self.local_expert_indices[0]:
                                                                      self.local_expert_indices[-1] + 1
                                                                ].contiguous()
            ep_mask = np.ones(self.ep_size, dtype=bool)
            ep_mask[curr_ep_rank] = False
            local_hot_mask = np.isin(self.local_expert_indices_np, self.hot_experts)
            ep_local_hot_mask = np.outer(ep_mask, local_hot_mask)
            num_global_cold_local_hot_tokens_per_local_expert[0, ep_local_hot_mask] = 0
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_cold_local_hot_tokens_per_rank = num_global_cold_local_hot_tokens_per_local_expert.sum(axis=2)

            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            self.output_splits = (
                num_global_cold_local_hot_tokens_per_rank[self.tp_rank]
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )

            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = (
                num_global_cold_local_hot_tokens_per_rank.sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            # Megatron will copy num_tokens_per_local_expert to cpu for GMM.
            # NPU GMM can support input splits tensor in device, so no need to copy.
            num_tokens_per_local_expert = num_global_cold_local_hot_tokens_per_local_expert.sum(dim=(0, 1))
        else:
            num_global_cold_local_hot_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

        self.padded_num_remote_hot_tokens_npu = torch.tensor(
            self.padded_num_remote_hot_tokens,
            dtype=torch.int32
        ).to(routing_map.device)
        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert_cpu = num_global_cold_local_hot_tokens_per_local_expert.view(
                -1, self.num_local_experts
            ).to(torch.device("cpu"), non_blocking=True)

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
        global_cold_local_hot_tokens_per_expert = self.preprocess(self.routing_map)

        sumnum_remote_hot_tokens = self.padded_num_remote_hot_tokens.sum().item()
        self.sumnum_cold_local_hot_tokens = routing_map.sum().item() - sumnum_remote_hot_tokens

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()

        # hostbound here
        self.sorted_routing_map = move_sorted_columns_to_end(routing_map, self.remote_hot_experts)
        sorted_probs = move_sorted_columns_to_end(probs, self.remote_hot_experts)

        event = torch.npu.current_stream().record_event()

        with torch.npu.stream(self.overlap_stream):
            global PREMUTE_FINISH_EVENT
            self.overlap_stream.wait_event(event)
            if self.config.moe_permute_fusion:
                permuted_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute(
                    hidden_states,
                    self.sorted_routing_map,
                    probs=sorted_probs,
                    num_out_tokens=self.num_out_tokens,
                    fused=self.config.moe_permute_fusion,
                )
            else:
                permuted_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute(
                    hidden_states,
                    self.sorted_routing_map,
                    probs=sorted_probs,
                    num_out_tokens=self.num_out_tokens,
                    fused=self.config.moe_permute_fusion,
                )
            PREMUTE_FINISH_EVENT = self.overlap_stream.record_event()

        permuted_local_tokens, permuted_remote_hot_tokens = CustomSliceFunction.apply(
            permuted_input_tokens, self.sumnum_cold_local_hot_tokens)
        permuted_local_probs, permuted_remote_hot_probs = CustomSliceFunction.apply(
            permuted_probs, self.sumnum_cold_local_hot_tokens)

        return ((permuted_local_tokens, permuted_remote_hot_tokens),
                (permuted_local_probs, permuted_remote_hot_probs), global_cold_local_hot_tokens_per_expert)

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
                global_input_tokens, self.tp_group,
                output_split_sizes=output_splits_tp.tolist() if output_splits_tp is not None else None,
                event=prob_comm_handle if prob_comm_handle else comm_handle, stream=torch.npu.current_stream()
            )
            if global_input_token_probs is not None:
                _, global_input_token_probs, prob_comm_handle = async_all_gather(
                    global_input_token_probs, self.tp_group,
                    output_split_sizes=output_splits_tp.tolist() if output_splits_tp is not None else None,
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

    def token_permute2(self, global_input_tokens, global_input_token_probs,
                       num_global_tokens_per_local_expert_cpu=None):
        num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert_cpu \
            if num_global_tokens_per_local_expert_cpu is not None else self.num_global_tokens_per_local_expert_cpu

        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if self.num_local_experts > 1:
            global_input_tokens, global_input_token_probs = sort_chunks_by_idxs(
                global_input_tokens,
                num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
                probs=global_input_token_probs,
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
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )

        return hidden_states

    def async_combine_comm(self, hidden_states, input_splits=None, output_splits=None, input_splits_tp=None,
                           wait_event=None):

        # Perform expert parallel AlltoAll communication
        output_splits = output_splits if output_splits is not None else self.input_splits
        input_splits = input_splits if input_splits is not None else self.output_splits
        input_splits_tp = input_splits_tp if input_splits_tp is not None else self.output_splits_tp

        if self.tp_size > 1:
            _, hidden_states, wait_event = async_reduce_scatter(
                hidden_states, self.tp_group,
                input_split_sizes=input_splits_tp.tolist() if input_splits_tp is not None else None,
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

    def backward_async_combine_comm(self, tokens_grad, input_splits=None, output_splits=None, output_splits_tp=None,
                                    wait_event=None):

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

    def token_unpermute2(
            self, permuted_local_hidden_states, remote_hot_hidden_states):
        merged_cold_hot_hidden_states = torch.cat([permuted_local_hidden_states, remote_hot_hidden_states], dim=0)
        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            merged_cold_hot_hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None

    def hot_expert_selection(self, num_global_tokens_per_expert, hot_expert_num):
        tokens_per_expert = num_global_tokens_per_expert.sum(axis=0).reshape(self.num_experts // self.num_local_experts,
                                                                             self.num_local_experts)
        tokens_per_ep = tokens_per_expert.sum(axis=1).tolist()

        hot_expert_list = []
        for _ in range(hot_expert_num):
            # Select the max expert on the max EP.
            max_ep_offset = np.argmax(tokens_per_ep)
            max_expert_offset = np.argmax(tokens_per_expert[max_ep_offset])
            hot_expert_id = max_ep_offset * self.num_local_experts + max_expert_offset
            local_selected_expert_per_ep = num_global_tokens_per_expert[:, hot_expert_id]
            tokens_sum_hot_expert_per_ep = tokens_per_expert[max_ep_offset][max_expert_offset]
            # Make the selected expert's token local.
            tokens_per_ep[max_ep_offset] -= tokens_sum_hot_expert_per_ep
            # Assigned to 0 to avoid being selected as hot expert again.
            tokens_per_expert[max_ep_offset][max_expert_offset] = 0
            tokens_per_ep += local_selected_expert_per_ep
            hot_expert_list.append(hot_expert_id)

        hot_expert = np.array(hot_expert_list)
        hot_expert.sort()

        return hot_expert
