# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import types
import copy
from copy import deepcopy
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
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


def moe_layer_init(self, config, submodules=None, layer_number=None):
    self.submodules = submodules
    super(MoELayer, self).__init__(config=config, layer_number=layer_number)
    self.router = TopKRouter(config=self.config)
    moe_experts_pipeline_degree = get_args().moe_experts_pipeline_degree
    if self.config.moe_grouped_gemm:
        if moe_experts_pipeline_degree == 0:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            expert = GroupedMLP(self.num_local_experts // moe_experts_pipeline_degree, self.config)
            self.experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(moe_experts_pipeline_degree)])
    else:
        if not isinstance(self.submodules, MLPSubmodules):
            raise TypeError("submodules should be instance of MLPSubmodules")
        self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
    if config.moe_token_dispatcher_type == "allgather":
        self.token_dispatcher = MoEAllGatherTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )
    elif config.moe_token_dispatcher_type == "alltoall":
        self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )
    else:
        raise ValueError(
            f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
        )

    return moe_layer_init


def moe_layer_init_wrapper(init_func):
    @wraps(init_func)
    def wrapper(*args, **kwargs):
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
                                                                linear_fc2=RowParallelLinear,),
                                          shared_expert=True)
            else:
                from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
                self.shared_experts = MLP(config, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                linear_fc2=RowParallelLinear,))

        self.moe_adaptive_recompute_activation = global_args.moe_adaptive_recompute_activation
        self.recompute_threshold = 0
        if hasattr(self.config, 'moe_token_dispatcher_type') and self.config.moe_token_dispatcher_type == 'allgather':
            self.moe_adaptive_recompute_activation_scale = global_args.moe_adaptive_recompute_activation_scale
            self.recompute_threshold = parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_data_parallel_world_size() * \
                self.config.moe_router_topk * self.moe_adaptive_recompute_activation_scale / self.config.num_moe_experts
            self.token_dispatcher.all_tokens_per_expert = None
        self.forward = types.MethodType(moe_adaptive_forward, self)

    return wrapper


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

    threshold = hidden_states.shape[0] * hidden_states.shape[1] * self.recompute_threshold
    moe_adaptive_recompute_activation_bool = self.moe_adaptive_recompute_activation and \
        (self.token_dispatcher.all_tokens_per_expert is None or torch.max(self.token_dispatcher.all_tokens_per_expert) > threshold)
    if self.moe_layer_recompute or moe_adaptive_recompute_activation_bool:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)
    return output, mlp_bias
