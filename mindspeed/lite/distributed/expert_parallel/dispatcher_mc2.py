# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from mindspeed.lite.distributed.dist_ops import gather_along_first_dim_expert_parallel
from mindspeed.lite.distributed.expert_parallel.utils import permute, unpermute
from mindspeed.lite.ops.grouped_matmul_mc2 import all2all_grouped_matmul, grouped_matmul_all2all


def ep_forward(ep_group, self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
               top_k_weights: torch.Tensor):
    hidden_states_shape = hidden_states.shape

    gate_weights_t = torch.stack([w.T for w in self.gate_weights], dim=0)
    up_weights_t = torch.stack([w.T for w in self.up_weights], dim=0)
    gate_up_weights = torch.cat([gate_weights_t, up_weights_t], dim=-1)
    down_weights = torch.stack([w.T for w in self.down_weights], dim=0)

    send_counts, recv_counts = dispatch_preprocess(self, ep_group, top_k_index)

    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index)

    hidden_states = all2all_grouped_matmul(hidden_states, gate_up_weights, ep_group, send_counts, recv_counts)
    gates, ups = torch.chunk(hidden_states, 2, dim=-1)
    hidden_states = self[0].act_fn(gates) * ups
    hidden_states = grouped_matmul_all2all(hidden_states, down_weights, ep_group, recv_counts, send_counts)

    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights)
    return hidden_states.view(*hidden_states_shape)


def dispatch_preprocess(module, ep_group, top_k_index):
    ep_size = torch.distributed.get_world_size(ep_group)

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=module.num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    send_counts = num_local_tokens_per_expert
    recv_counts = num_global_tokens_per_expert.reshape(ep_size, module.num_global_experts)[:,
                  module.local_expert_indices[0]: module.local_expert_indices[-1] + 1].reshape(-1)
    return send_counts, recv_counts
