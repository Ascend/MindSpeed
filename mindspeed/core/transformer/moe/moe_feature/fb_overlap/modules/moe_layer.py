# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.transformer.moe.moe_feature import TopKRouter


class MindSpeedFbOverlapMoELayer(BaseMoELayer):
    def __init__(
        self,
        config,
        submodules=None,
        layer_number=None,
        pg_collection=None,
        is_mtp_layer=False,
        name=None,
    ):
        self.submodules = submodules
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        # shared_expert two param mutual conversion
        if config.n_shared_experts:
            config.moe_shared_expert_intermediate_size = config.n_shared_experts * (
                config.moe_ffn_hidden_size if config.moe_ffn_hidden_size is not None else config.ffn_hidden_size
            )
        super().__init__(
            config,
            layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.moe_layer_recompute = False

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        if not hasattr(self.config, 'shared_expert_gate'):
            self.config.shared_expert_gate = None

        # Initialize experts
        if not self.config.moe_grouped_gemm:
            raise ValueError("use fb overlap should open moe_grouped_gemm")
        self.experts = self.submodules.experts(
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
            name=(name + ".experts") if name is not None else None,
        )

        # Initialize token dispatcher
        # ``alltoall`` is a dispatcher identifier, not a credential.
        if self.config.moe_token_dispatcher_type == 'alltoall':  # nosec B105
            from .token_dispatcher import MindSpeedMOEAlltoAllFbOverlapTokenDispatcher

            self.token_dispatcher = MindSpeedMOEAlltoAllFbOverlapTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise AssertionError('currently fb overlap only support alltoall token dispatcher')

        # Initialize shared experts through Megatron 0.18's builder protocol.
        if self.use_shared_expert:
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            # fb overlap set shared expert overlap by default
            self.shared_expert_overlap = True

        if getattr(get_args(), "print_expert_load", False) or getattr(get_args(), "enable_expert_placement", False):
            if get_args().moe_tp_extend_ep:
                self.ep_world_size = parallel_state.get_tensor_and_expert_parallel_world_size()
            else:
                self.ep_world_size = parallel_state.get_expert_model_parallel_world_size()

            self.expert_placement_init(self.ep_world_size)

    def forward(self, hidden_states):
        # FB overlap will not call forward for entire MoE Layer
        pass
