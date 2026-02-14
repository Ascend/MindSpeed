# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from torch_npu.utils.collect_env import get_cann_version
import torch.nn.functional as F
from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state, 
    tensor_parallel, 
    permute, 
    unpermute, 
    bias_geglu_impl,
    bias_gelu_impl,
    bias_swiglu_impl,
    sort_chunks_by_idxs,
    reduce_scatter_to_sequence_parallel_region,
    MoEAlltoAllTokenDispatcher,
    gather_from_sequence_parallel_region,
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    set_tensor_grad_fn_sequence_sr,
    get_capacity,
    )
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func, 
    async_comm_sort_chunks_by_idxs,
    )
from mindspeed.core.transformer.moe.moe_feature.overlap.unpermute_without_activation import UnpermuteWithoutActivation
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import (
    async_all_to_all, 
    async_alltoall_with_backward,
    async_reduce_scatter
    )

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


def is_less_or_equal_rc2_cann_version():
    '''
    check Ascend CANN version.
    '''
    cann_starts_with = ('8.0.RC1', '8.0.RC2')
    cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
                '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
    cann_version = get_cann_version()
    return cann_version in cann_all or cann_version.startswith(cann_starts_with)

cann_version_check = is_less_or_equal_rc2_cann_version()


class MoEAlltoAllSeqOverLapDispatcher:
    """
    The legacy implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    We've kept the old version of the Mindspeed MoEAlltoAlloverlap here.

    Note: This class is a modification of the MoEAlltoAllTokenDispatcher from version 0.8.0, and 
    called as 'MoEAlltoAllSEQTokenDispatcher' after Megatron core_r0.9.0.
    """

    def __init__(self, num_local_experts, local_expert_indices, config):
        """
        Initialize the AlltoAllSeq token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
    
        self.num_local_experts = num_local_experts
        self.config = config
        self.local_expert_indices = local_expert_indices
        # use MOEAlltoAllSEQTokenDispatcher to init
        super().__init__(num_local_experts, local_expert_indices, config)
        if self.config.moe_tp_extend_ep:
            from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher
            self.disaptor = MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher(num_local_experts, local_expert_indices, config)
        else:
            self.disaptor = MoEAlltoAllSEQTokenDispatcher(num_local_experts, local_expert_indices, config)

    def preprocess_overlap(self, routing_map):

        num_tokens_per_local_expert = self.disaptor.preprocess(routing_map)
        self.num_global_tokens_per_local_expert = self.disaptor.num_global_tokens_per_local_expert
        self.input_splits = self.disaptor.input_splits
        self.output_splits = self.disaptor.output_splits
        self.num_out_tokens = self.disaptor.num_out_tokens
        self.num_global_tokens_per_local_expert_cpu = self.disaptor.num_global_tokens_per_local_expert_cpu
        self.comm_stream = (
            self.disaptor.comm_stream
            if (self.config.moe_tp_extend_ep and hasattr(self.disaptor, 'comm_stream'))
            else None
        )
        self.cuda_sync_point = self.disaptor.cuda_sync_point
        return num_tokens_per_local_expert

    def token_permutation(
        self, 
        hidden_states: torch.Tensor, 
        probs: torch.Tensor, 
        routing_map: torch.Tensor, 
        shared_experts, 
        save_tensors, 
        moe_ctx=None
    ):
        """
        Dispatch tokens to local experts using AlltoAllSeq communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            routing_map (torch.Tensor): Mapping of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            shared_experts: A Mindspeed shared_experts Model.
            save_tensors (List): Save Tensors During permutation and unpermutation
                for MoELayerOverlapAll2AllSeq's recompute.
            moe_ctx: Config settings from MoELayerOverlapAll2All.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        self.hidden_shape = hidden_states.shape
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for routing map"

        # Permutation 1: input to AlltoAll input
        def alltoall_token_permutation1(hidden_states, routing_map, permuted_probs):
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
            tokens_per_expert = self.preprocess_overlap(routing_map)
            if not self.config.moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)
            self.hidden_shape_before_permute = hidden_states.shape
            if self.cuda_sync_point == "before_permutation_1":
                torch.cuda.current_stream().synchronize()
            (
                permutated_local_input_tokens,
                permuted_probs,
                self.reversed_local_input_permutation_mapping,
            ) = permute(hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens,
                        fused=self.config.moe_permute_fusion)

            return permutated_local_input_tokens, permuted_probs, tokens_per_expert

        (permutated_local_input_tokens, permuted_probs, tokens_per_expert), *_ = forward_func(
                                                                alltoall_token_permutation1, (hidden_states, routing_map, probs))

        # permute 1
        save_tensors.append(permutated_local_input_tokens)
        save_tensors.append(permuted_probs)
        ep_group = parallel_state.get_expert_model_parallel_group()
        if self.config.moe_tp_extend_ep:
            ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()

        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            ep_group,
        )

        _, global_probs, permute1_probs_handle = async_all_to_all(
            permuted_probs,
            self.output_splits,
            self.input_splits,
            ep_group
        )

        # shared experts compute.
        if shared_experts is not None:
            if self.config.moe_zero_memory != "disable":
                (share_experts_output), *_ = forward_func(shared_experts, (hidden_states, moe_ctx))
            else:
                (share_experts_output), *_ = forward_func(shared_experts, (hidden_states))
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                share_experts_graph, share_experts_output, rs_shared_experts_handle = async_reduce_scatter(share_experts_output, parallel_state.get_tensor_model_parallel_group(),
                                                                                                        event=permute1_ep_all_to_all_handle, stream=torch.npu.default_stream())
                share_experts_output = (share_experts_graph, share_experts_output, rs_shared_experts_handle)
        else:
            share_experts_output = None

        permute1_ep_all_to_all_handle.wait()
        permuted_probs.untyped_storage().resize_(0)
        permutated_local_input_tokens.untyped_storage().resize_(0)

        def alltoall_token_permutation2(global_input_tokens, global_probs, permute1_probs_handle=None):
            # Permutation 2: Sort tokens by local expert.
            if self.num_local_experts > 1:
                global_input_tokens, global_probs = async_comm_sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert_cpu.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    prob_handle=permute1_probs_handle
                )
            else:
                # Avoid memory released before used.
                global_input_tokens, global_probs = global_input_tokens.clone(), global_probs.clone()

            # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
            # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
            if (not self.config.moe_tp_extend_ep and
                    parallel_state.get_tensor_model_parallel_world_size() > 1 and
                    self.config.moe_grouped_gemm):
                global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                    global_input_tokens
                )
            if self.cuda_sync_point == "before_finish":
                torch.cuda.current_stream().synchronize()

            return global_input_tokens, global_probs

        save_tensors.append(self.num_global_tokens_per_local_expert_cpu)
        moe_ctx.sort_input_by_local_experts = self.sort_input_by_local_experts

        # token premute2 input
        (global_input_tokens, global_probs), global_input_tokens_detach, global_probs_detach, _ = forward_func(alltoall_token_permutation2,
                                                                        (global_input_tokens, global_probs, permute1_probs_handle))

        save_tensors.append(global_input_tokens_detach)
        save_tensors.append(global_input_tokens)
        save_tensors.append(global_probs_detach)
        save_tensors.append(global_probs)
        global_input_tokens_detach.untyped_storage().resize_(0)
        global_probs_detach.untyped_storage().resize_(0)
        return share_experts_output, global_input_tokens, tokens_per_expert, global_probs


    def token_unpermutation(
        self, 
        hidden_states: torch.Tensor, 
        bias: torch.Tensor = None,
        save_tensors = None
    ):
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
        """

        #def alltoall_token_unpermutation1(hidden_states):
        assert bias is None, "Bias is not supported in MoEAlltoAllSeqTokenDispatcher"
        def alltoall_token_unpermutation1(hidden_states):
        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
            if not self.config.moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states)

            # Unpermutation 2: expert output to AlltoAll input.
            if self.num_local_experts > 1:
                hidden_states, _ = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                    self.restore_output_by_local_experts,
                )
            else:
                # Avoid memory released before used.
                hidden_states = hidden_states.clone()
            return hidden_states
        hidden_states, hidden_states_detach = forward_func(alltoall_token_unpermutation1, hidden_states)
        save_tensors.append(hidden_states_detach)
        hidden_states_detach.untyped_storage().resize_(0)
        #unpermute1_graph
        save_tensors.append(hidden_states)
        ep_group = parallel_state.get_expert_model_parallel_group()
        if self.config.moe_tp_extend_ep:
            ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]

        permutated_local_input_tokens = tensor_parallel.all_to_all(
            ep_group,
            hidden_states,
            self.input_splits,
            self.output_splits,
        )
        hidden_states.untyped_storage().resize_(0)
        
        def alltoall_token_unpermutation2(permutated_local_input_tokens):
            output = unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                restore_shape=self.hidden_shape_before_permute,
                routing_map=self.routing_map,
                fused=self.config.moe_permute_fusion
            )

            # Perform tensor parallel AlltoAll communication.
            # output: [S*B, H/TP] -> [S*B/TP, H]
            if not self.config.moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
                output = tensor_parallel.all_to_all_hp2sp(output)

            # Reshape the output tensor.
            output = output.view(self.hidden_shape)
            return output
        
        output, unpermute2_input_detach = forward_func(alltoall_token_unpermutation2, permutated_local_input_tokens)
        save_tensors.append(unpermute2_input_detach)
        unpermute2_input_detach.untyped_storage().resize_(0)
        permutated_local_input_tokens.untyped_storage().resize_(0)
        return output


class MoEAllGatherOverLapDispatcher:
    """
    AllGather Based Token dispatcher With Overlap.
    Note that, in core_r0.10.0, the allgather spans the communication domain of TP*EP:
    """

    def __init__(self, num_local_experts, local_expert_indices, config, pg_collection):
        """
        Initialize the AlltoAllSeq token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
    
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert!"
        self.config = config
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index!"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs = None

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where
        # each element is True if it's between the local_expert_indices. Only useful when cross
        # device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

        # use MoEAllGatherTokenDispatcher to init
        super().__init__(num_local_experts, local_expert_indices, config, pg_collection)

    def token_permutation(
        self, 
        global_routing_map_tuple: tuple, 
        global_probs_tuple: tuple, 
        global_hidden_states_tuple: tuple,
    ):
        """
        Dispatch tokens to local experts using AllGather communication.

        Args:
            global_routing_map_tuple (tuple): Include routing_map (torch.Tensor) and gr_handle for control async communication.
                routing_map: 2D tensor [S/TP*B, num_experts], representing token assignment to
                global experts.
            global_probs_tuple (tuple): Include global_probs (torch.Tensor) and gp_handle for control async communication.
                probs: 2D tensor [S/TP*B, num_experts]. Each row of probs contains
                the probility distribution across `topk` experts for one local token.
            global_hidden_states_tuple (tuple): Include global_hidden_states (torch.Tensor) and ghs_handle for control async communication.
                hidden_states: 3D tensor [S/TP, B, H]. Input tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - local expert map.
                - expert tokens reversed map.
        """
        global_routing_map, gr_handle = global_routing_map_tuple
        global_probs, gp_handle = global_probs_tuple
        global_hidden_states, ghs_handle = global_hidden_states_tuple
        tokens_per_expert = None

        if (self.config.tensor_model_parallel_size > 1) or (
                self.config.expert_model_parallel_size > 1
        ):
            
            with (torch.no_grad()):

                gr_handle.wait()
            
            gp_handle.wait()
            # masked_select -> reshape
        self.local_probs = global_probs[
            :, self.local_expert_indices[0]:self.local_expert_indices[-1] + 1
        ].contiguous()
        self.local_map = global_routing_map[
            :, self.local_expert_indices[0]:self.local_expert_indices[-1] + 1
        ].contiguous()
        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()    
        ghs_handle.wait()
        self.hidden_shape_before_permute = global_hidden_states.shape

        (permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping) = permute(
            global_hidden_states, self.local_map
        )
        return (
            permuted_local_hidden_states,
            tokens_per_expert,
            self.local_map,
            self.reversed_local_input_permutation_mapping
        )


    def token_unpermutation(
        self, 
        hidden_states: torch.Tensor, 
        bias: torch.Tensor = None, 
        reversed_local_input_permutation_mapping: torch.Tensor = None
        ):
        # Stage1: unpermute the tokens and bias locally respectively.

        permuted_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        hidden_states = hidden_states * permuted_probs.unsqueeze(-1)
        unpermuted_local_hidden = unpermute(
            hidden_states,
            reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
        )

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            bias = bias * permuted_probs.unsqueeze(-1)
            unpermuted_local_bias = unpermute(
                bias,
                reversed_local_input_permutation_mapping,
                restore_shape=self.hidden_shape_before_permute,
            )

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        if (self.tp_size > 1) or (
                self.ep_size > 1
        ):
            if self.add_bias:
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region(
                    output_bias_total, group=parallel_state.get_expert_tensor_and_model_parallel_group()
                )
                output_bias_total = (output_bias_total / self.tp_size)

        output_total = output_total.view(self.hidden_shape_before_permute)
        if self.add_bias:
            output_bias_total = output_bias_total.view(self.hidden_shape_before_permute)

        return output_total, output_bias_total


class MoEAlltoAllOverLapDispatcher(MoEAlltoAllTokenDispatcher):
    """
    An AlltoAll-based token dispatcher with overlap.
    Same as MoEAlltoAllSeqOverLapDispatcher, also support moe-zero-memory.
    The dispatcher support shared expert overlap method. 

    Note that in the overlap method of different branches, the shared experts' forward 
    function are defined in AlltoAllOverlapMoeLayer (without moe_shared_expert_overlap) 
    or MoEAlltoAllOverLapDispatcher (with moe_shared_expert_overlap).
    """

    def __init__(self, num_local_experts, local_expert_indices, config, pg_collection):
        """
        Initialize the AlltoAll overlap token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
    
        self.num_local_experts = num_local_experts
        self.config = config
        self.local_expert_indices = local_expert_indices
        # use MOEAlltoAllTokenDispatcher to init
        super().__init__(num_local_experts, local_expert_indices, config, pg_collection)


    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.config.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts], number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts].
            self.num_global_tokens_per_local_expert_cpu = torch.full(
                (self.num_experts * self.tp_size,), self.capacity, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.config.moe_expert_capacity_factor is not None:
            # Drop tokens to capacity, no padding.
            # A synchronization is needed before the first
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
                :, :, self.local_expert_indices[0]:self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
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
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1)).to(
                torch.device("cpu"), non_blocking=True
            )
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            ).to(torch.device("cpu"), non_blocking=True)

        return num_tokens_per_local_expert

    def pre_forward_comm(self, hidden_states):
        """
        All Gather for SP before forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        Unlike megatron, the backward compute graph is rearrange, so torch < 2.2 can use
        this part.
        """
        self.shared_experts.gate_score = None
        assert self.shared_experts.config.moe_shared_expert_overlap
        assert self.shared_experts.cached_output is None
        self.shared_experts.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.shared_experts.stream):
            if self.shared_experts.use_shared_expert_gate:
                logits = torch.nn.functional.linear(hidden_states, self.shared_experts.gate_weight)
                self.shared_experts.gate_score = torch.nn.functional.sigmoid(logits)
            if self.shared_experts.config.sequence_parallel:
                cached_fc1_input = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=True
                )
            else:
                cached_fc1_input = copy_to_tensor_model_parallel_region(hidden_states)
        return cached_fc1_input 

    def linear_fc1_forward_and_act(self, cached_fc1_input):
        """
        Do Linear FC1 and activation function forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.shared_experts.config.moe_shared_expert_overlap
        assert cached_fc1_input is not None

        with torch.cuda.stream(self.shared_experts.stream):
            # [s, b, 4 * h/p]
            self.cached_fc1_output, bias_parallel = self.shared_experts.linear_fc1(cached_fc1_input)
            if self.shared_experts.config.bias_activation_fusion:
                if self.shared_experts.activation_func == F.gelu:
                    if self.shared_experts.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            self.cached_fc1_output, bias_parallel
                        )
                    else:
                        assert self.shared_experts.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.shared_experts.activation_func == F.silu and self.shared_experts.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        self.cached_fc1_output,
                        bias_parallel,
                        self.shared_experts.config.activation_func_fp8_input_store,
                    )

                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    intermediate_parallel = self.cached_fc1_output + bias_parallel
                if self.shared_experts.config.gated_linear_unit:
                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.shared_experts.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(self.cached_fc1_output)
                else:
                    intermediate_parallel = self.shared_experts.activation_func(self.cached_fc1_output)
            cached_fc2_input = intermediate_parallel
        return cached_fc2_input

    def token_permutation(
        self, 
        hidden_states: torch.Tensor, 
        probs: torch.Tensor, 
        routing_map: torch.Tensor, 
        save_tensors, 
        moe_ctx=None
    ):
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            routing_map (torch.Tensor): Mapping of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            save_tensors (List): Save Tensors During permutation and unpermutation
                for MoELayerOverlapAll2AllSeq's recompute and backward.
            moe_ctx: Config settings from MoELayerOverlapAll2All.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - Permuted router probs for local experts.
        """
        self.hidden_shape = hidden_states.shape
        self.routing_map = routing_map
        self.cached_fc1_input = None
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for routing map"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"

        # Permutation 1: input to AlltoAll input
        def alltoall_token_permutation1(hidden_states, routing_map):

            tokens_per_expert = self.preprocess(self.routing_map)
            tokens_per_expert = tokens_per_expert.to('npu', non_blocking=True)
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

            # Megatron shared experts overlap settings.
            # With Megatron shared experts overlap, the shared_experts move to dispatcher.
            # Other conditions, the shared_experts belong to moe_layer.
            if moe_ctx.shared_expert_overlap:
                self.cached_fc1_input = self.pre_forward_comm(hidden_states.view(self.hidden_shape))

            self.hidden_shape_before_permute = hidden_states.shape   
    
            if self.cuda_sync_point == "before_permutation_1":
                torch.cuda.current_stream().synchronize()

            (
                permutated_local_input_tokens,
                permuted_probs,
                self.reversed_local_input_permutation_mapping,
            ) = permute(hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens, fused=self.config.moe_permute_fusion)

            return permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping, tokens_per_expert, self.cached_fc1_input

        (permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping, tokens_per_expert, self.cached_fc1_input), *_ = forward_func(
                                                                alltoall_token_permutation1, (hidden_states, routing_map))

        # permute 1.
        save_tensors.append(permutated_local_input_tokens)
        save_tensors.append(permuted_probs)
        if self.shared_experts is not None:
            moe_ctx.share_experts_graph_list.append(self.cached_fc1_input)

        # Perform expert parallel AlltoAll communication.
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )

        _, global_probs, permute1_probs_handle = async_all_to_all(
            permuted_probs,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )

        #Shared_experts overlap compute and save cached_fc1_input for zm1 backward recompute.
        if self.shared_experts is not None:
            self.cached_fc2_input, cached_fc1_input_detach = forward_func(\
                self.linear_fc1_forward_and_act, (self.cached_fc1_input))
            if moe_ctx.config.moe_zero_memory == "level1" and not moe_ctx.is_only_recompute_activation:
                moe_ctx.cached_fc1_output = self.cached_fc1_output
                moe_ctx.cached_fc1_output.untyped_storage().resize_(0)
                moe_ctx.cached_fc1_input = self.cached_fc1_input
                #Avoid cached_fc1_input memory blast when TP=1 with zm1.
                if parallel_state.get_expert_tensor_parallel_world_size() > 1:
                    moe_ctx.cached_fc1_input.untyped_storage().resize_(0)
            #fc1 input
            moe_ctx.share_experts_graph_list.append(cached_fc1_input_detach)

        permute1_probs_handle.wait()
        permute1_ep_all_to_all_handle.wait()
        permuted_probs.untyped_storage().resize_(0)
        permutated_local_input_tokens.untyped_storage().resize_(0)

        if self.tp_size > 1:
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens,
                group=self.tp_group,
                output_split_sizes=(
                    self.output_splits_tp.tolist() if self.output_splits_tp is not None else None
                ),
            )
            global_probs = gather_from_sequence_parallel_region(
                global_probs,
                group=self.tp_group,
                output_split_sizes=(
                    self.output_splits_tp.tolist() if self.output_splits_tp is not None else None
                ),
            )
    
        permutated_local_input_tokens.untyped_storage().resize_(0)

        def alltoall_token_permutation2(global_input_tokens, global_probs):

            # Permutation 2: Sort tokens by local expert.
            if self.num_local_experts > 1:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert_cpu.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                )

            if self.cuda_sync_point == "before_finish":
                torch.cuda.current_stream().synchronize()

            return global_input_tokens, global_probs

        # token premute2 input.
        (global_input_tokens, global_probs), global_input_tokens_detach, global_probs_detach = forward_func(alltoall_token_permutation2,
                                                                        (global_input_tokens, global_probs))
        save_tensors.append(global_input_tokens_detach)
        save_tensors.append(global_input_tokens)
        save_tensors.append(global_probs_detach)
        save_tensors.append(global_probs)
        global_input_tokens_detach.untyped_storage().resize_(0)
        global_probs_detach.untyped_storage().resize_(0)
        return global_input_tokens, tokens_per_expert, global_probs

    def linear_fc2_forward(self, overlapped_comm_output=None):
        """
        Do Linear FC2 forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.shared_experts.config.moe_shared_expert_overlap
        assert self.cached_fc2_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.shared_experts.stream):
            # [s, b, h]
            self.cached_fc2_output, _ = self.shared_experts.linear_fc2(self.cached_fc2_input)

    def post_forward_comm(self):
        """
        Reduce scatter for SP after forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_output is not None
        with torch.cuda.stream(self.shared_experts.stream):
            if self.config.sequence_parallel:
                shared_expert_output = reduce_scatter_to_sequence_parallel_region(
                    self.cached_fc2_output
                )
            else:
                shared_expert_output = reduce_from_tensor_model_parallel_region(
                    self.cached_fc2_output
                )
            self.cached_fc2_output = None
            set_tensor_grad_fn_sequence_sr(shared_expert_output, torch.iinfo(torch.int).max)
        return shared_expert_output

    def token_unpermutation(
        self, 
        hidden_states: torch.Tensor, 
        bias: torch.Tensor = None, 
        moe_ctx=None
    ):
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).
            moe_ctx: Config settings from MoELayerOverlapAll2All.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter.
        # Unpermutation 2: expert output to AlltoAll input.
        if self.num_local_experts > 1:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )
        if self.tp_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states,
                group=self.tp_group,
                input_split_sizes=(
                    self.output_splits_tp.tolist() if self.output_splits_tp is not None else None
                ),
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        #use async_alltoall_with_backward to overlap forward alltoall.
        permutated_local_input_tokens, handle = async_alltoall_with_backward(
            self.ep_group,
            hidden_states,
            self.input_splits,
            self.output_splits,
        )

        #share expert.
        if self.shared_experts is not None:
            self.linear_fc2_forward(permutated_local_input_tokens)
            if moe_ctx.shared_expert_overlap:
                if moe_ctx.config.moe_zero_memory == "level1" and not moe_ctx.is_only_recompute_activation:
                    moe_ctx.shared_act_out = self.cached_fc2_input
                    moe_ctx.shared_act_out.untyped_storage().resize_(0)
            shared_expert_output = self.post_forward_comm()

        handle.wait()
        # Unpermutation 2: expert output to AlltoAll input.
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=None,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion
        )
        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        if self.shared_experts is not None:
            with torch.cuda.stream(self.shared_experts.stream):
                if self.shared_experts.use_shared_expert_gate:
                    assert self.gate_score is not None
                    shared_expert_output = shared_expert_output * self.gate_score
                    self.gate_score = None
            torch.cuda.current_stream().wait_stream(self.shared_experts.stream)
            output += shared_expert_output

        return output
