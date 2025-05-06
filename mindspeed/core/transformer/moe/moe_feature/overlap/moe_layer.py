# Copyright (c) 2025, Huawei Technologies. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_all2all import MoELayerOverlapAllToAll
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_allgather import MoELayerOverlapAllGather
from mindspeed.core.transformer.moe.moe_feature import (
    MegatronModule,
    parallel_state,
    TopKRouter,
    MLP,
    MLPSubmodules,
    SharedExpertMLP)


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.
        In "AllToAll_Seq" Dispatcher, when "moe_tp_extend_ep" is set, the number of experts is split instead of
        the H dimension (Which is a bit like Megatron "AllToAll" Dispatcher after core_r0.9.0.).
    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        layer_number (int):The layer number for the MoE layer.
    """

    def __init__(self, config, layer_number: int = None):
        MegatronModule.__init__(self, config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert self.config.num_moe_experts % (self.expert_parallel_size * tp_size) == 0
        # adjust the local expert split logic
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size // tp_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts * tp_size +
            parallel_state.get_tensor_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class AlltoAllSeqOverlapMoeLayer(BaseMoELayer):
    """
    Sets the MoE_layer when "moe-alltoall-overlap-comm" is used.
    """
    def __init__(self, config, submodules=None, layer_number=None):
        """
        "moe-alltoall-overlap-comm" only supported "moe_grouped_gemm".
        """
        self.submodules = submodules
        self.config = config
        super().__init__(config, layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(config=self.config)

        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor, \
            MindSpeedOverLapGmmExperts
        # Initialize experts
        if not self.config.moe_grouped_gemm:
            raise ValueError(
                f"use '--moe-alltoall-overlap-comm' should open '--moe-grouped-gemm'."
            )
        else:
            self.experts = MindSpeedOverLapGmmExperts(self.num_local_experts, self.config)

        # Initialize token dispatcher
        self.token_dispatcher = MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
            if self.shared_expert_overlap:
                raise ValueError(
                    f"use tp_extend_ep not support shared_expert_overlap."
                )
            # In 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed version to 'with_shared_expert'.
            self.shared_experts.with_shared_expert = True

    def forward(self, hidden_states):
        return MoELayerOverlapAllToAll.apply(hidden_states, self.config, self)


class AllGatherOverlapMoeLayer(BaseMoELayer):
    '''
    Sets the MoE_layer when "moe-allgather-overlap-comm" is used.
    '''
    def __init__(self, config, submodules=None, layer_number=None):
        """
        "moe-allgather-overlap-comm" only supported "moe_grouped_gemm".
        """
        self.submodules = submodules
        self.config = config
        super().__init__(config, layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(config=self.config)

        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedMOEAllGatherOverLapDispatcherAdaptor, \
            MindSpeedOverLapGmmExperts
        # Initialize experts
        if self.config.moe_grouped_gemm:
            self.experts = MindSpeedOverLapGmmExperts(self.num_local_experts, self.config)
        else:
            raise ValueError(
                f"use '--moe-allgather-overlap-comm' should open '--moe_grouped_gemm'."
            )

        # Initialize token dispatcher
        self.token_dispatcher = MindSpeedMOEAllGatherOverLapDispatcherAdaptor(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = SharedExpertMLP(self.config, self.submodules.shared_experts)
            if self.shared_expert_overlap:
                raise ValueError(
                    f"use tp_extend_ep not support shared_expert_overlap."
                )
            # In 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed version to 'with_shared_expert'.
            self.shared_experts.with_shared_expert = True
        self.token_dispatcher.all_tokens_per_expert = None

    def forward(self, hidden_states):
        return MoELayerOverlapAllGather.apply(hidden_states, self.config, self)