# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch
from mindspeed.core.transformer.moe.moe_feature import (
    MegatronModule,
    parallel_state,
    TopKRouter,
    MLP,
    MLPSubmodules,
    SharedExpertMLP,
    ModuleSpec,
    build_module,
    tensor_parallel
    )


class MSBaseMoELayer(MegatronModule):
    """Base class for a mixture of experts layer adjust for use tp_extend_ep.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
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

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class All2AllSeqTpExtendEpMoELayerImpl(MSBaseMoELayer):
    def __init__(self, config, submodules=None, layer_number=None):
        """
        All2AllSeq use tp_extend_ep
        router and shared_expert use megatron original
        Args:
            config: TransformerConfig
            submodules: model specs
            layer_number: number of layer
        """
        self.submodules = submodules
        self.config = config
        super().__init__(config, layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(config=self.config)

        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher, \
            MindSpeedTpExtendEpGmmExperts
        # Initialize experts
        if not self.config.moe_grouped_gemm:
            raise ValueError(
                f"use tp_extend_ep should open moe_grouped_gemm"
            )
        self.experts = MindSpeedTpExtendEpGmmExperts(self.num_local_experts, self.config)

        # Initialize token dispatcher
        self.token_dispatcher = MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                raise ValueError(
                    f"use tp_extend_ep not support shared_expert_overlap"
                )

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            if self.use_shared_expert:
                if not hasattr(self, 'comm_stream'):
                    self.comm_stream = torch.cuda.Stream()
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    share_experts_output = self.shared_experts(hidden_states)

            (dispatched_input, tokens_per_expert, permuted_probs) = (
                self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
            )
            expert_output, mlp_bias = self.experts(
                dispatched_input, tokens_per_expert, permuted_probs
            )
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + share_experts_output
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

