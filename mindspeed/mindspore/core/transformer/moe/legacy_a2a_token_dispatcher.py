# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import mindspore
from torch_npu.utils.collect_env import get_cann_version
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.transformer.moe.moe_utils import moe_gather, moe_scatter, permute, unpermute
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel, \
    reduce_scatter_to_sequence_parallel_region
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
from mindspeed.core.transformer.moe.comm_utils import (async_reduce_scatter,
                                                       async_all_gather, reduce_scatter, all_gather)
from mindspeed.mindspore.core.transformer.moe.moe_layer_overlap_all2all import forward_func
from mindspeed.core.transformer.moe.unpermute_without_activation import UnpermuteWithoutActivation
from mindspeed.core.transformer.moe.moe_utils import (AG_SHARED_EXPERTS_INPUTS,
                                                      permute_with_ep, unpermute_with_ep)
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
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor, shared_experts, save_tensors, shared_expert_gate, moe_ctx=None
):
    moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    if moe_hierarchical_alltoallv:
        ep_group = parallel_state.get_expert_model_parallel_group()
        _, indices, indices_handle = async_all_gather(indices, group=ep_group)
        indices_handle.wait()
        save_tensors.append(indices)
        _, hidden_states_ep, hidden_states_ep_handle = async_all_gather(hidden_states, group=ep_group)
    else:
        indices_ep, hidden_states_ep, hidden_states_ep_handle = None, None, None
        save_tensors.append(indices_ep)

    if moe_hierarchical_alltoallv:
        tokens_per_expert = self.preprocess(indices, hidden_states)
    else:
        tokens_per_expert = self.preprocess(indices)
    save_tensors.append(hidden_states_ep)
    #, indices, *args
    def alltoall_token_permutation1(hidden_states):
        if moe_hierarchical_alltoallv:
            _, self.probs, probs_handle = async_all_gather(self.probs, group=ep_group)
            hidden_states_ep_handle.wait()
            # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
            self.hidden_shape_before_permute = hidden_states.shape
            # Permutation 1: input to AlltoAll input
            if self.cuda_sync_point == "before_permutation_1":
                mindspore.runtime.current_stream().synchronize()
            probs_handle.wait()
            self.probs = mindspore.ops.stop_gradient(self.probs)
            self.probs.requires_grad = True
            save_tensors.append(self.probs)
            permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute_with_ep(
                hidden_states, indices, probs=self.probs, topk=self.router_topk,
                gb_inputs_splits=self.input_splits_tp_ep,
            )
            self.permuted_probs = permuted_probs
        else:
            if get_args().moe_experts_pipeline_degree:
                tokens_per_expert = tokens_per_expert.cpu()

            # Flatten the input tensor
            # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

            # Perform tensor parallel AlltoAll communication
            # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
            if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

            # Permutation 1: input to AlltoAll input
            self.hiddden_shape_before_permute = hidden_states.shape
            if self.cuda_sync_point == "before_permutation_1":
                mindspore.runtime.current_stream().synchronize()
            scores_ep = None
            save_tensors.append(scores_ep)
            permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
                hidden_states,
                indices,
                num_out_tokens=self.num_out_tokens,
                padded_mode=self.drop_and_pad,
            )
        return permutated_local_input_tokens

    input_hidden_states = hidden_states_ep if moe_hierarchical_alltoallv else hidden_states
    permutated_local_input_tokens, *_, vjp_alltoall_token_permutation1 = forward_func(alltoall_token_permutation1,
                                                                                      input_hidden_states)

    # permute 1
    save_tensors.append(permutated_local_input_tokens)

    # Perform expert parallel AlltoAll communication
    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()

    # Perform expert parallel AlltoAll communication
    if self.cuda_sync_point == "before_ep_alltoall":
        mindspore.runtime.current_stream().synchronize()
    if moe_hierarchical_alltoallv:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            tp_group,
        )
    else:
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            ep_group,
        )

    # shared experts
    if shared_experts is not None:
        def shared_experts_func(hidden_states):
            output, bias = shared_experts(hidden_states)
            return output, bias
        (share_experts_output, _), *_, vjp_shared_experts = forward_func(shared_experts_func, hidden_states)
        if parallel_state.get_tensor_model_parallel_world_size() > 1 and shared_expert_gate is None:
            share_experts_graph, share_experts_output, rs_shared_experts_handle = async_reduce_scatter(share_experts_output, parallel_state.get_tensor_model_parallel_group(),
                                                                                                       event=permute1_ep_all_to_all_handle, stream=mindspore.runtime.default_stream())
            share_experts_output = (share_experts_graph, share_experts_output, rs_shared_experts_handle)
        if shared_expert_gate is not None:
            with torch.enable_grad():
                # tp not support shared expert gate for now
                if parallel_state.get_tensor_model_parallel_world_size() > 1:
                    share_experts_output = reduce_scatter_to_sequence_parallel_region(share_experts_output)
                share_experts_output = torch.nn.functional.sigmoid(shared_expert_gate(hidden_states)) * share_experts_output
    else:
        share_experts_output = None

    if permute1_ep_all_to_all_handle is not None:
        permute1_ep_all_to_all_handle.wait()
        del permutated_local_input_tokens

    def alltoall_token_permutation2(global_input_tokens):
        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                if self.comm_stream is not None:
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
        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        need_tp_comm = (not get_args().moe_tp_extend_ep and
                        parallel_state.get_tensor_model_parallel_world_size() > 1 and
                        self.config.moe_grouped_gemm) and get_args().moe_experts_pipeline_degree == 0
        if need_tp_comm:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )
        if self.cuda_sync_point == "before_finish":
            mindspore.runtime.current_stream().synchronize()

        return global_input_tokens

    # token 重排2 input
    (global_input_tokens), global_input_tokens_detach, vjp_alltoall_token_permutation2 = forward_func(alltoall_token_permutation2,
                                                                     global_input_tokens)
    save_tensors.append(global_input_tokens_detach)
    save_tensors.append(global_input_tokens)
    del global_input_tokens_detach

    return share_experts_output, global_input_tokens, tokens_per_expert, vjp_shared_experts, vjp_alltoall_token_permutation1, vjp_alltoall_token_permutation2


def alltoall_token_unpermutation_new(
        self, hidden_states, bias, save_tensors
):
    moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv

    def alltoall_token_unpermutation1(hidden_states):
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1 and get_args().moe_experts_pipeline_degree == 0:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states)

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                hidden_states = unpermute(
                    hidden_states, self.reversed_global_input_permutation_mapping,
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
    if get_args().moe_experts_pipeline_degree:
        with torch.enable_grad():
            hidden_states = alltoall_token_unpermutation1(hidden_states)
        save_tensors.append(hidden_states)
    else:
        hidden_states, unpermute1_input_detach, vjp_alltoall_token_unpermutation1 = forward_func(alltoall_token_unpermutation1, hidden_states)
        save_tensors.append(unpermute1_input_detach)
        save_tensors.append(hidden_states)
        del unpermute1_input_detach

    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    # Perform expert parallel AlltoAll communication
    # hidden_states: [SEQL, H] -> [SEQL, H/TP]
    if moe_hierarchical_alltoallv:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            self.input_splits,
            self.output_splits,
            tp_group
        )
    else:
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            self.input_splits,
            self.output_splits,
            ep_group
        )
    if handle is not None:
        handle.wait()
        del hidden_states

    def alltoall_token_unpermutation2(permutated_local_input_tokens, probs):
        # Unpermutation 1: AlltoAll output to output
        if get_args().moe_zero_memory != "disable":
            output = UnpermuteWithoutActivation.apply(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                self.probs
            )
        else:
            if moe_hierarchical_alltoallv:
                unpermute_with_ep_input_tensors_list = [permutated_local_input_tokens,
                                                        self.reversed_local_input_permutation_mapping,
                                                        self.permuted_probs]
                output = unpermute_with_ep(
                    unpermute_with_ep_input_tensors_list,
                    restore_shape=self.hidden_shape_before_permute,
                    probs=self.probs,
                    topk=self.router_topk
                )
            else:
                output = unpermute(
                    permutated_local_input_tokens,
                    self.reversed_local_input_permutation_mapping,
                    probs=self.probs,
                    padded_mode=self.drop_and_pad,
                    restore_shape=self.hiddden_shape_before_permute,
                )
        if moe_hierarchical_alltoallv:
            return output
        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output

    output, unpermute2_input_detach, _, vjp_alltoall_token_unpermutation2 = forward_func(alltoall_token_unpermutation2, (permutated_local_input_tokens, self.probs))
    save_tensors.append(unpermute2_input_detach)
    should_resize = not self.drop_and_pad and not moe_hierarchical_alltoallv and \
                    not get_args().use_fused_moe_token_permute_and_unpermute or get_args().moe_zero_memory != "disable"
    if should_resize:
        del unpermute2_input_detach
    save_tensors.append(output)

    if moe_hierarchical_alltoallv:
        ep_group = parallel_state.get_expert_model_parallel_group()
        _, output, output_handle = async_reduce_scatter(output, group=ep_group)
        output_handle.wait()
        output = output.view(self.hidden_shape)
    return output, None, vjp_alltoall_token_unpermutation1, vjp_alltoall_token_unpermutation2
