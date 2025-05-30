# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import types
from copy import deepcopy
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules, MLP

from mindspeed.auto_settings.module.black.patch.hccl_operator import MOEOrMLPEndOp, MOEOrMLPStartOp
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
        global_args = get_args()
        init_func(*args, **kwargs)
        self = args[0]

        # In 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed version to 'with_shared_expert'.
        if self.use_shared_expert:
            self.shared_experts.with_shared_expert = True
        
        # In 0.10.0, 'MoEAlltoAllSEQTokenDispatcher' no longer has'add_bias' attribute. 
        # To use the two types of share_expert(Megatron and MindSpeed), this parameter is introduced temporarily.
        if self.config.add_bias_linear and self.config.moe_token_dispatcher_type != 'alltoall':
            self.token_dispatcher.add_bias = self.config.add_bias_linear
        else:
            self.token_dispatcher.add_bias = None

        self.moe_alltoall_overlap_comm = global_args.moe_alltoall_overlap_comm
        self.moe_allgather_overlap_comm = global_args.moe_allgather_overlap_comm

        self.moe_adaptive_recompute_activation = global_args.moe_adaptive_recompute_activation
        self.recompute_threshold = 0
        if hasattr(self.config, 'moe_token_dispatcher_type') and self.config.moe_token_dispatcher_type == 'allgather':
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
        if args.prof_file and args.num_experts > 1:
            hidden_states = MOEOrMLPStartOp.apply(hidden_states)
            activation_func1 = torch.nn.Softplus()
            hidden_states = activation_func1(hidden_states)

        probs, routing_map = self.router(hidden_states)
        if args.n_shared_experts or args.moe_shared_expert_intermediate_size:
            if not hasattr(self, 'comm_stream'):
                self.comm_stream = torch.cuda.Stream()
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                share_experts_output = self.shared_experts(hidden_states)
        (dispatched_input, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if args.n_shared_experts or args.moe_shared_expert_intermediate_size:
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            output = output + share_experts_output

        if args.prof_file and args.num_experts > 1:
            activation_func2 = torch.nn.Softshrink()
            output = activation_func2(output)
            output = MOEOrMLPEndOp.apply(output)
        return output, mlp_bias

    threshold = hidden_states.shape[0] * hidden_states.shape[1] * self.recompute_threshold
    moe_adaptive_recompute_activation_bool = self.moe_adaptive_recompute_activation and \
        (self.token_dispatcher.all_tokens_per_expert is None or torch.max(self.token_dispatcher.all_tokens_per_expert) > threshold)
    if self.moe_layer_recompute or moe_adaptive_recompute_activation_bool:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)
    return output, mlp_bias


def zero_memory_SharedExpertMlp_forward(self, hidden_states, moe_ctx):
    """Shared expert forward function with zero_memory."""
    output, _ = MLP.forward(self, hidden_states, moe_ctx)
    if self.use_shared_expert_gate:
        logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
        gate_score = torch.nn.functional.sigmoid(logits)
        output = output * gate_score
    return output
