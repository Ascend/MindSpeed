# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import mindspore
from torch_npu.utils.collect_env import get_cann_version
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.transformer.moe.moe_utils import permute, unpermute, sort_chunks_by_idxs, get_capacity
from megatron.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
from mindspeed.core.transformer.moe.comm_utils import (async_reduce_scatter,
                                                       async_all_gather)
from mindspeed.mindspore.core.transformer.moe.moe_layer_overlap_all2all import forward_func
from mindspeed.core.transformer.moe.unpermute_without_activation import UnpermuteWithoutActivation
from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS
from mindspeed.mindspore.core.transformer.moe.comm_utils import async_all_to_all


def is_less_or_equal_rc2_cann_version():
    cann_starts_with = ('8.0.RC1', '8.0.RC2')
    cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
                '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
    cann_version = get_cann_version()
    return cann_version in cann_all or cann_version.startswith(cann_starts_with)


cann_version_check = is_less_or_equal_rc2_cann_version()


def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
    # use 0.7.0 implement for better performance
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    if self.drop_and_pad:
        # probs: [num_experts, capacity]
        self.capacity = self.probs.size(1)
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long,
            device=torch.cuda.current_device()
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        # Token drop but no pad. A synchronization is needed before the first
        # permutation to get the `num_out_tokens` CPU value.
        self.num_out_tokens = num_local_tokens_per_expert.sum()
        self.cuda_sync_point = "before_permutation_1"
    elif ep_size > 1:
        # Token dropless and enable ep. A synchronization is needed before expert parallel
        # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
        self.cuda_sync_point = "before_ep_alltoall"
    else:
        # Token dropless and no ep. A synchronization is needed before the token_permutation()
        # function returns to get the `tokens_per_expert` CPU value.
        self.cuda_sync_point = "before_finish"

    if ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .numpy()
        )
        num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
            num_local_tokens_per_expert
        ).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                                                  :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
                                                  ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1).numpy()
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
        if not hasattr(self, 'comm_stream'):
            self.comm_stream = mindspore.runtime.Stream()
        self.comm_stream.wait_stream(mindspore.runtime.current_stream())
        with mindspore.runtime.StreamCtx(self.comm_stream):
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

    return num_tokens_per_local_expert


def alltoall_token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
):
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    tokens_per_expert = self.preprocess(indices)

    # Flatten the input tensor
    # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Perform tensor parallel AlltoAll communication
    # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

    # Permutation 1: input to AlltoAll input
    self.hiddden_shape_before_permute = hidden_states.shape
    if self.cuda_sync_point == "before_permutation_1":
        mindspore.runtime.current_stream().synchronize()
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states,
        indices,
        num_out_tokens=self.num_out_tokens,
        padded_mode=self.drop_and_pad,
    )

    if get_args().moe_bmm_mc2:
        return permutated_local_input_tokens, tokens_per_expert

    # Perform expert parallel AlltoAll communication
    if self.cuda_sync_point == "before_ep_alltoall":
        mindspore.runtime.current_stream().synchronize()
    global_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_expert_model_parallel_group(),
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
    )

    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
    if self.num_local_experts > 1:
        if not self.drop_and_pad:
            mindspore.runtime.current_stream().wait_stream(self.comm_stream)
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

    # Perform tensor parallel All-Gather on the hidden dimension to obtain the input tokens.
    # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and self.config.moe_grouped_gemm:
        global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
            global_input_tokens
        )
    if self.cuda_sync_point == "before_finish":
        mindspore.runtime.current_stream().synchronize()

    return global_input_tokens, tokens_per_expert


def alltoall_token_permutation_new(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        shared_experts,
        save_tensors,
        shared_expert_gate,
        moe_ctx=None):
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    self.routing_map = routing_map
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert routing_map.dim() == 2, "Expected 2D tensor for routing map"

    tokens_per_expert = self.preprocess(routing_map)

    # Permutation 1: input to AlltoAll input
    def alltoall_token_permutation1(hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)
        self.hidden_shape_before_permute = hidden_states.shape

        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()
        permutated_local_input_tokens, _, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            routing_map,
            num_out_tokens=self.num_out_tokens,
        )
        return permutated_local_input_tokens

    permutated_local_input_tokens, *_, vjp_alltoall_token_permutation1 = forward_func(
        alltoall_token_permutation1, hidden_states)
    # permute 1
    save_tensors.append(permutated_local_input_tokens)

    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
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

    # shared experts
    if shared_experts is not None:
        if get_args().moe_zero_memory != "disable":
            (share_experts_output, _), *_, vjp_shared_experts = forward_func(shared_experts, (hidden_states, moe_ctx))
        else:
            (share_experts_output, _), *_, vjp_shared_experts = forward_func(shared_experts, (hidden_states))
        if shared_expert_gate is not None:
            with torch.enable_grad():
                # tp not support shared expert gate for now
                if parallel_state.get_tensor_model_parallel_world_size() > 1:
                    share_experts_output = reduce_scatter_to_sequence_parallel_region(share_experts_output)
                share_experts_output = torch.nn.functional.sigmoid(
                    shared_expert_gate(hidden_states)) * share_experts_output
    else:
        share_experts_output = None

    permute1_ep_all_to_all_handle.wait()
    permutated_local_input_tokens.untyped_storage().resize_(0)

    def alltoall_token_permutation2(global_input_tokens):
        # Permutation 2: Sort tokens by local expert.
        if self.num_local_experts > 1:
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
            )

        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if (not get_args().moe_tp_extend_ep and
                parallel_state.get_tensor_model_parallel_world_size() > 1 and
                self.config.moe_grouped_gemm):
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )
        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens

    save_tensors.append(self.num_global_tokens_per_local_expert_cpu)
    moe_ctx.sort_input_by_local_experts = self.sort_input_by_local_experts

    # token 重排2 input
    (global_input_tokens), global_input_tokens_detach, vjp_alltoall_token_permutation2 = forward_func(alltoall_token_permutation2,
                                                                     global_input_tokens)
    save_tensors.append(global_input_tokens_detach)
    save_tensors.append(global_input_tokens)
    global_input_tokens_detach.untyped_storage().resize_(0)

    return share_experts_output, global_input_tokens, tokens_per_expert, None, vjp_shared_experts, vjp_alltoall_token_permutation1, vjp_alltoall_token_permutation2


def alltoall_token_unpermutation_new(
        self, hidden_states, bias, save_tensors
):
    def alltoall_token_unpermutation1(hidden_states):
        assert bias is None, "Bias is not supported in MoEAlltoAllSeqTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states)

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )
        return hidden_states

    hidden_states, unpermute1_input_detach, vjp_alltoall_token_unpermutation1 = forward_func(alltoall_token_unpermutation1, hidden_states)
    save_tensors.append(unpermute1_input_detach)
    save_tensors.append(hidden_states)
    unpermute1_input_detach.untyped_storage().resize_(0)

    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()
    # Perform expert parallel AlltoAll communication
    # hidden_states: [SEQL, H] -> [SEQL, H/TP]
    _, permutated_local_input_tokens, handle = async_all_to_all(
        hidden_states,
        self.input_splits,
        self.output_splits,
        ep_group
    )
    handle.wait()
    hidden_states.untyped_storage().resize_(0)

    def alltoall_token_unpermutation2(permutated_local_input_tokens, probs):
        # Unpermutation 1: AlltoAll output to output
        if get_args().moe_zero_memory != "disable":
            output = UnpermuteWithoutActivation.apply(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                self.hidden_shape_before_permute,
                self.probs,
                self.routing_map,
            )
        else:
            output = unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                probs=self.probs,
                restore_shape=self.hidden_shape_before_permute,
                routing_map=self.routing_map,
            )

        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output

    output, unpermute2_input_detach, _, vjp_alltoall_token_unpermutation2 = forward_func(alltoall_token_unpermutation2, (permutated_local_input_tokens, self.probs))
    save_tensors.append(unpermute2_input_detach)

    return output, None, vjp_alltoall_token_unpermutation1, vjp_alltoall_token_unpermutation2
