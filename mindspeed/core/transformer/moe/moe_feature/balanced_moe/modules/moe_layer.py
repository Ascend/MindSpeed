# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch_npu
from megatron.core.transformer import build_module
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn.parameter import Parameter

from mindspeed.core.transformer.moe.moe_feature.balanced_moe.modules.token_dispatcher import (
    MoEBalancedAlltoAllTokenDispatcher
)


class SharedParamsForHotExperts:
    def __init__(self, hot_experts_module, grad_w1_hot_shape, grad_w2_hot_shape):
        self.shared_weight1 = Parameter(
            torch.empty_like(hot_experts_module.weight1.data)
        )
        self.shared_weight1.is_hot_experts = True
        self.shared_weight1.requires_grad = True
        self.shared_weight1.gmm_weight = True

        self.shared_weight2 = Parameter(
            torch.empty_like(hot_experts_module.weight2.data)
        )
        self.shared_weight2.is_hot_experts = True
        self.shared_weight2.requires_grad = True
        self.shared_weight2.gmm_weight = True
        self.grad_w1_buffer = torch.empty_like(
            hot_experts_module.weight1.data, dtype=torch.float32, device="cuda", memory_format=torch.contiguous_format
        ).view(grad_w1_hot_shape)
        self.grad_w2_buffer = torch.empty_like(
            hot_experts_module.weight2.data, dtype=torch.float32, device="cuda", memory_format=torch.contiguous_format
        ).view(grad_w2_hot_shape)

    def register_shared_weight(self, hot_experts_module):
        hot_experts_module.weight1 = self.shared_weight1
        hot_experts_module.weight2 = self.shared_weight2


_GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS = None


def init_global_shared_params_for_hot_experts(hot_experts_module, grad_w1_hot_shape, grad_w2_hot_shape):
    global _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS
    if _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS is None:
        _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS = SharedParamsForHotExperts(hot_experts_module, grad_w1_hot_shape, grad_w2_hot_shape)


def get_shared_params_for_hot_experts():
    global _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS
    assert _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS is not None
    return _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS


def get_shared_grad_for_hot_experts():
    global _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS
    assert _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS is not None
    return (_GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS.grad_w1_buffer, _GLOBAL_SHARED_PARAMS_FOR_HOT_EXPERTS.grad_w2_buffer)


class BalancedMoELayer(BaseMoELayer):
    """Balanced Mixture of experts Layer **currently only supports no token dropping**.
    """

    def __init__(
            self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        # shared_expert two param mutual conversion
        if config.n_shared_experts:
            config.moe_shared_expert_intermediate_size = config.n_shared_experts * (
                config.moe_ffn_hidden_size if config.moe_ffn_hidden_size is not None else config.ffn_hidden_size)
        super(BalancedMoELayer, self).__init__(config=config, layer_number=layer_number)

        self.moe_layer_recompute = False

        # Initialize router
        self.router = TopKRouter(config=self.config)

        if not hasattr(self.config, 'shared_expert_gate'):
            self.config.shared_expert_gate = None

        # Initialize experts
        if not self.config.moe_grouped_gemm:
            raise ValueError(
                f"use fb overlap should open moe_grouped_gemm"
            )
        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)
        self.hot_experts = build_module(self.submodules.experts, self.config.balanced_moe_hot_expert_num, self.config)
        self.hot_experts.is_hot_experts = True
        self.hot_experts.weight1.is_hot_experts = True
        self.hot_experts.weight2.is_hot_experts = True

        # Initialize token dispatcher
        if self.config.moe_token_dispatcher_type == 'alltoall':
            self.token_dispatcher = MoEBalancedAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise AssertionError('Currently, --balanced-moe-experts only support alltoall token dispatcher')

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            # fb overlap set shared expert overlap by default
            self.shared_expert_overlap = True

        self.moe_layer_recompute = config.moe_layer_recompute
        self.num_hot_experts = self.config.balanced_moe_hot_expert_num

        # Broadcast and MLP synchronization.
        self.hot_expert_finish_events = [torch_npu.npu.Event() for _ in range(self.num_hot_experts)]
        self.expert_broadcast_streams = [torch_npu.npu.Stream() for _ in range(self.num_hot_experts)]
        self.hot_expert_broadcast_handles = [[] for _ in range(self.num_hot_experts)]
        # Seperate weight1/2 handles.
        self.hot_expert_inter_ep_grad_reduce_handles = [[None for _ in range(self.num_hot_experts)] for _ in range(2)]
        self.params = [
            self.num_local_experts,
            self.local_expert_indices,
            self.expert_broadcast_streams,
            self.hot_expert_finish_events,
            self.hot_expert_broadcast_handles,
            self.hot_expert_inter_ep_grad_reduce_handles
        ]

        # Hook related.
        self.hot_experts_list = [None for _ in range(self.num_hot_experts)]

        # Register balanced expert reduce hook to each local parameter.

        grad_w1_local_shape = (self.num_local_experts, self.experts.config.hidden_size, -1)
        grad_w1_hot_shape = (self.num_hot_experts, self.hot_experts.config.hidden_size, -1)
        self.experts.weight1.grad_local_shape = grad_w1_local_shape
        self.experts.weight1.grad_hot_shape = grad_w1_hot_shape
        grad_w2_local_shape = (self.num_local_experts, -1, self.experts.config.hidden_size)
        grad_w2_hot_shape = (self.num_hot_experts, -1, self.hot_experts.config.hidden_size)
        self.experts.weight2.grad_local_shape = grad_w2_local_shape
        self.experts.weight2.grad_hot_shape = grad_w2_hot_shape

        self.hot_experts.weight1.untyped_storage_size = self.hot_experts.weight1.untyped_storage().size()
        self.hot_experts.weight2.untyped_storage_size = self.hot_experts.weight2.untyped_storage().size()

        init_global_shared_params_for_hot_experts(self.hot_experts, grad_w1_hot_shape, grad_w2_hot_shape)
        self.hot_experts.weight1 = None
        self.hot_experts.weight2 = None

        # Should be set to False to avoid M-Core allocating ParamGradBuffer and DP gradient accumulator hook.
        self.hot_experts.requires_grad_(False)

    def forward(self, hidden_states):
        # FB overlap will not call forward for entire MoE Layer
        pass
