# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from torch_npu.utils.collect_env import get_cann_version
from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state, 
    tensor_parallel, 
    permute, 
    unpermute, 
    sort_chunks_by_idxs,
    reduce_scatter_to_sequence_parallel_region,
    MoEAlltoAllSEQTokenDispatcher
    )
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func
)
from mindspeed.core.transformer.moe.moe_feature.overlap.unpermute_without_activation import UnpermuteWithoutActivation
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import async_all_to_all, async_reduce_scatter

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
        self.comm_stream = self.disaptor.comm_stream if self.config.moe_tp_extend_ep else None
        return num_tokens_per_local_expert

    def token_permutation(
        self, 
        hidden_states: torch.Tensor, 
        probs: torch.Tensor, 
        routing_map: torch.Tensor, 
        shared_experts, 
        save_tensors, 
        shared_expert_gate, 
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
            shared_expert_gate: Use shared_expert_gate to replace reduce_scatter 
                in shared_expert with TP=1.
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
            tokens_per_expert = tokens_per_expert.to('npu', non_blocking=True)
            if self.cuda_sync_point == "before_permutation_1":
                torch.cuda.current_stream().synchronize()
            (
                permutated_local_input_tokens,
                permuted_probs,
                self.reversed_local_input_permutation_mapping,
            ) = permute(hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens)

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
            if shared_expert_gate is not None:
                with torch.enable_grad():
                    # tp not support shared expert gate for now.
                    if parallel_state.get_tensor_model_parallel_world_size() > 1:
                        share_experts_output = reduce_scatter_to_sequence_parallel_region(share_experts_output)
                    share_experts_output = torch.nn.functional.sigmoid(shared_expert_gate(hidden_states)) * share_experts_output
        else:
            share_experts_output = None

        permute1_probs_handle.wait()
        permute1_ep_all_to_all_handle.wait()
        permuted_probs.untyped_storage().resize_(0)
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
        (global_input_tokens, global_probs), global_input_tokens_detach, global_probs_detach = forward_func(alltoall_token_permutation2,
                                                                        (global_input_tokens, global_probs))

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

        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
        )

        # Perform tensor parallel AlltoAll communication.
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if not self.config.moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor.
        output = output.view(self.hidden_shape)

        return output


class MoEAllGatherOverLapDispatcher:
    """
    AllGather Based Token dispatcher With Overlap.
    Note that, in core_r0.10.0, the allgather spans the communication domain of TP*EP:
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
        super().__init__(num_local_experts, local_expert_indices, config)

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
            output_total = reduce_scatter_to_sequence_parallel_region(
                    output_total, group=self.tp_ep_group
                )
            if self.add_bias:
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region(
                    output_bias_total, group=self.tp_ep_group
                )
                output_bias_total = (output_bias_total / self.tp_size)

        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total.view(self.hidden_shape)

        return output_total, output_bias_total
