# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

from mindspeed.core.transformer.moe.moe_feature import (
    MegatronModule,
    parallel_state,
    TopKRouter,
    TransformerConfig,
    ModuleSpec,
    tensor_parallel,
)


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class TSBaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ):
        MegatronModule.__init__(self, config)
        self.config = config
        self.pg_collection = pg_collection
        self.is_mtp_layer = is_mtp_layer
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = parallel_state.get_expert_model_parallel_rank() * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
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


class AlltoAllMC2MoeLayer(TSBaseMoELayer):
    """Mixture of experts Layer
        **currently only supports no token dropping**.
        This layer is adjusted to use fused MC2 kernal.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
    ):
        self.submodules = submodules
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer
        )
        self.moe_layer_recompute = config.recompute_granularity == 'selective' and "moe" in config.recompute_modules

        # Initialize router
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)

        # Initialize token dispatcher
        from mindspeed.core.transformer.moe.moe_feature.adaptor import (
            MoEAlltoAllSEQMC2TokenDispatcherAdaptor,
            MoEAlltoAllMC2TokenDispatcherAdaptor,
        )

        if config.moe_token_dispatcher_type == "alltoall_seq":  # nosec B105
            self.token_dispatcher = MoEAlltoAllSEQMC2TokenDispatcherAdaptor(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":  # nosec B105
            self.token_dispatcher = MoEAlltoAllMC2TokenDispatcherAdaptor(
                self.num_local_experts, self.local_expert_indices, config=self.config, pg_collection=pg_collection
            )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")

        # Initialize experts
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedGmmMC2Experts

        self.experts = MindSpeedGmmMC2Experts(self.num_local_experts, self.config, pg_collection=pg_collection)

        # Initialize shared experts
        if self.use_shared_expert:
            self.config.moe_shared_expert_overlap = True
            # default open shared_expert_overlap.
            # NOTE: in Megatron 0.18 the shared_experts spec is a keyword-only builder
            # partial (see moe_module_specs._build_shared_experts), which cannot go
            # through build_module. Call it directly following MoELayer's protocol.
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            # if self.shared_expert_overlap:
            self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if self.training and self.config.tensor_model_parallel_size > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            # do p and gmm weight 1
            # padding_mask is not consumed by TopKRouter, do not forward it
            probs, routing_map = self.router(hidden_states)
            (dispatched_input, tokens_per_expert, permuted_probs, dispatcher) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )

            # No more needs expert cal. Only do a act cal.
            up_out, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs, dispatcher)
            # do gmm weight2 & up
            output, mlp_bias = self.token_dispatcher.token_unpermutation(up_out, mlp_bias)

            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
