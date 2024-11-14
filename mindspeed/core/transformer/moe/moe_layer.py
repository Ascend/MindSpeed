# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import types
from copy import deepcopy
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import MoELayerOverlapAll2All
from mindspeed.core.transformer.moe.moe_layer_overlap_allgather import MoELayerOverlapAllGather


def base_moe_init_wrapper(init_func):
    @wraps(init_func)
    def base_moe_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        if global_args.moe_tp_extend_ep:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            assert self.config.num_moe_experts % (self.expert_parallel_size * tp_size) == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size // tp_size
            local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * self.num_local_experts * tp_size + \
                    parallel_state.get_tensor_model_parallel_rank() * self.num_local_experts
            )
            self.local_expert_indices = [
                local_expert_indices_offset + i for i in range(self.num_local_experts)
            ]
            assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))

    return base_moe_init


def moe_layer_init_wrapper(init_func):
    @wraps(init_func)
    def moe_layer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        self.moe_alltoall_overlap_comm = global_args.moe_alltoall_overlap_comm
        self.moe_allgather_overlap_comm = global_args.moe_allgather_overlap_comm

        if global_args.n_shared_experts:
            config = deepcopy(self.config)
            config.ffn_hidden_size = global_args.n_shared_experts * self.config.ffn_hidden_size
            if self.moe_allgather_overlap_comm or self.moe_alltoall_overlap_comm:
                from mindspeed.core.transformer.moe.layers import ColumnParallelLinear, RowParallelLinear
                self.shared_experts = MLP(config, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                     linear_fc2=RowParallelLinear, ),
                                          shared_expert=True)
            else:
                from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
                self.shared_experts = MLP(config, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                     linear_fc2=RowParallelLinear, ))

        self.moe_adaptive_recompute_activation = global_args.moe_adaptive_recompute_activation
        if hasattr(global_args, 'moe_token_dispatcher_type') and global_args.moe_token_dispatcher_type == 'allgather':
            self.moe_adaptive_recompute_activation_scale = global_args.moe_adaptive_recompute_activation_scale
            self.recompute_threshold = parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_data_parallel_world_size() * \
                self.config.moe_router_topk * self.moe_adaptive_recompute_activation_scale / self.config.num_moe_experts
            self.token_dispatcher.all_tokens_per_expert = None
        self.forward = types.MethodType(moe_adaptive_forward, self)

    return moe_layer_init


def moe_adaptive_forward(self, hidden_states: torch.Tensor):
    if self.moe_alltoall_overlap_comm:
        return MoELayerOverlapAll2All.apply(hidden_states, self)
    if self.moe_allgather_overlap_comm:
        return MoELayerOverlapAllGather.apply(hidden_states, self)

    def custom_forward(hidden_states):
        args = get_args()
        scores, indices = self.router(hidden_states)
        if args.n_shared_experts:
            if not hasattr(self, 'comm_stream'):
                self.comm_stream = torch.cuda.Stream()
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                share_experts_output, share_experts_bias = self.shared_experts(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, scores, indices
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if args.n_shared_experts:
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            output = output + share_experts_output
            if self.token_dispatcher.add_bias:
                mlp_bias = mlp_bias + share_experts_bias
        return output, mlp_bias

    if self.moe_adaptive_recompute_activation:
        threshold = hidden_states.shape[0] * hidden_states.shape[1] * self.recompute_threshold
        if self.token_dispatcher.all_tokens_per_expert is None or torch.max(
                self.token_dispatcher.all_tokens_per_expert) > threshold:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)
    return output, mlp_bias