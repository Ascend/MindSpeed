# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies. All rights reserved.

import torch
from mindspeed.core.transformer.moe.moe_feature import (
    parallel_state, 
    tensor_parallel, 
    permute, 
    )
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    forward_func, 
    async_comm_sort_chunks_by_idxs,
    )
from mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils import (
    async_all_to_all,
    async_reduce_scatter
    )


def token_permutation(
    self, 
    hidden_states: torch.Tensor, 
    probs: torch.Tensor, 
    routing_map: torch.Tensor, 
    shared_experts, 
    save_tensors, 
    moe_ctx=None
):
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

    save_tensors.append(torch.from_numpy(self.num_global_tokens_per_local_expert_cpu))
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