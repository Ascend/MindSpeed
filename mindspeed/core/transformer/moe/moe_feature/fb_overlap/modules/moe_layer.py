# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import types
from copy import deepcopy

from mindspeed.core.transformer.moe.moe_feature import TopKRouter, BaseMoELayer, build_module


class MindSpeedFbOverlapMoELayer(BaseMoELayer):
    def __init__(self, config, submodules=None, layer_number=None):
        self.submodules = submodules
        # shared_expert two param mutual conversion
        if config.n_shared_experts:
            config.moe_shared_expert_intermediate_size = config.n_shared_experts * config.moe_ffn_hidden_size
        super(MindSpeedFbOverlapMoELayer, self).__init__(config, layer_number)

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

        # Initialize token dispatcher
        if self.config.moe_token_dispatcher_type == 'alltoall':
            from .token_dispatcher import MindSpeedMOEAlltoAllFbOverlapTokenDispatcher
            self.token_dispatcher = MindSpeedMOEAlltoAllFbOverlapTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise AssertionError('currently fb overlap only support alltoall token dispatcher')

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            # fb overlap set shared expert overlap by default
            self.shared_expert_overlap = True

    def forward(self, hidden_states):
        # FB overlap will not call forward for entire MoE Layer
        pass