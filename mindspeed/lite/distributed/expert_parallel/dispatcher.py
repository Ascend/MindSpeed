# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from mindspeed.lite.distributed.expert_parallel.utils import permute, unpermute
from mindspeed.lite.distributed.dist_ops import all_to_all, gather_along_first_dim_expert_parallel
from mindspeed.lite.ops.grouped_matmul import eager_grouped_matmul, fused_grouped_matmul


def ep_forward(ep_group, fused, self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
               top_k_weights: torch.Tensor):
    hidden_states_shape = hidden_states.shape
    weights = (self.gate_weights, self.up_weights, self.down_weights)
    act_fn = self[0].act_fn
    num_global_experts = self.num_global_experts
    expert_ids_per_ep_rank = self.expert_ids_per_ep_rank

    hidden_states = dispatch_mlp_combine(ep_group, fused, hidden_states, top_k_index, top_k_weights, weights, act_fn,
                                         num_global_experts, expert_ids_per_ep_rank)

    return hidden_states.view(*hidden_states_shape)


def dispatch_mlp_combine(ep_group, fused, hidden_states, top_k_index, top_k_weights, weights, act_fn,
                         num_global_experts,
                         expert_ids_per_ep_rank):
    gate_weights, up_weights, down_weights = weights
    experts_computation = fused_experts_computation if fused else eager_experts_computation

    # MoE preprocess to get local/global indices and all2all split sizes
    permute_indices, split_sizes = dispatch_preprocess(ep_group, top_k_index, num_global_experts,
                                                       expert_ids_per_ep_rank)
    # Alltoall dispatch --> MLP computation --> Alltoall combine
    hidden_states, unpermute_indices = alltoall_dispatch(ep_group, hidden_states, top_k_index, permute_indices,
                                                         split_sizes)
    hidden_states = experts_computation(hidden_states, permute_indices[0], gate_weights, up_weights, down_weights,
                                        act_fn)
    hidden_states = alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes)
    return hidden_states


def dispatch_preprocess(ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank):
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    # [EP*E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[:,
                                         local_experts_start_id: local_experts_end_id]
    # [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0).to(torch.device("cpu"),
                                                                                    non_blocking=True)
    # [E] --> [EP, local_E] --> [EP]
    input_split = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                                                                 non_blocking=True)
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True)
    # [EP, local_E] --> [E*select]
    global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    return (num_tokens_per_local_expert, global_indices), (input_split, output_splits)


def alltoall_dispatch(ep_group, hidden_states, top_k_index, indices, split_sizes):
    local_indices, global_indices = indices
    input_split, output_splits = split_sizes

    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index)
    torch.npu.current_stream().synchronize()
    hidden_states = all_to_all(ep_group, hidden_states, output_splits.numpy(), input_split.numpy())
    hidden_states, unpermute_indices2 = permute(hidden_states, global_indices)

    return hidden_states, (unpermute_indices1, unpermute_indices2)


def eager_experts_computation(hidden_states, split_list, gate_weights, up_weights, down_weights, act_fn):
    gates = eager_grouped_matmul(hidden_states, split_list, *gate_weights)
    ups = eager_grouped_matmul(hidden_states, split_list, *up_weights)
    act = act_fn(gates) * ups
    hidden_states = eager_grouped_matmul(act, split_list, *down_weights)
    return hidden_states


def fused_experts_computation(hidden_states, split_list, gate_weights, up_weights, down_weights, act_fn):
    gates = fused_grouped_matmul(hidden_states, split_list, *gate_weights)
    ups = fused_grouped_matmul(hidden_states, split_list, *up_weights)
    act = act_fn(gates) * ups
    hidden_states = fused_grouped_matmul(act, split_list, *down_weights)
    return hidden_states


def alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes):
    unpermute_indices1, unpermute_indices2 = unpermute_indices
    input_split, output_splits = split_sizes
    hidden_states = unpermute(hidden_states, unpermute_indices2)
    hidden_states = all_to_all(ep_group, hidden_states, input_split.numpy(), output_splits.numpy())
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights)
    return hidden_states
