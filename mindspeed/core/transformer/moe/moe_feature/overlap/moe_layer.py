# Copyright (c) 2025, Huawei Technologies. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from megatron.core import utils
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_all2allseq import MoELayerOverlapAllToAllSeq
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_all2all import MoELayerOverlapAllToAll
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_allgather import MoELayerOverlapAllGather
from mindspeed.core.transformer.moe.moe_feature import (
    MegatronModule,
    TopKRouter,
    MLPSubmodules,
    MegatronBaseMoeLayer,
    TransformerConfig,
)


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.
        In "AllToAll_Seq" Dispatcher, when "moe_tp_extend_ep" is set, the number of experts is split instead of
        the H dimension (Which is a bit like Megatron "AllToAll" Dispatcher after core_r0.9.0.).
    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        layer_number (int):The layer number for the MoE layer.
    """

    def __init__(
        self,
        config,
        layer_number: int = None,
        pg_collection=None,
        is_mtp_layer: bool = False,
    ):
        MegatronModule.__init__(self, config)
        self.config = config
        self.pg_collection = pg_collection
        self.is_mtp_layer = is_mtp_layer
        self.expert_parallel_size = utils.get_pg_size(pg_collection.ep)
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_tp_extend_ep:
            tp_size = utils.get_pg_size(pg_collection.expt_tp)
            assert self.config.num_moe_experts % (self.expert_parallel_size * tp_size) == 0
            # adjust the local expert split logic
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size // tp_size
            local_expert_indices_offset = (
                utils.get_pg_rank(pg_collection.ep) * self.num_local_experts * tp_size
                + utils.get_pg_rank(pg_collection.expt_tp) * self.num_local_experts
            )
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = utils.get_pg_rank(pg_collection.ep) * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
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


def _get_experts_builder_kwargs(submodules):
    if submodules is None:
        return {}
    experts_builder = submodules.experts
    if isinstance(experts_builder, partial):
        return dict(experts_builder.keywords or {})
    return {}


def _get_te_grouped_mlp_submodules(experts_builder):
    """Find the TE GroupedMLPSubmodules through MindSpeed's nested builders."""
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP

    pending = [experts_builder]
    visited = set()
    found_te_builder = False
    grouped_mlp_submodules = None

    while pending:
        builder = pending.pop()
        builder_id = id(builder)
        if builder_id in visited:
            continue
        visited.add(builder_id)

        if builder is TEGroupedMLP:
            found_te_builder = True
            continue
        if not isinstance(builder, partial):
            continue

        if builder.func is TEGroupedMLP:
            found_te_builder = True
        submodules = (builder.keywords or {}).get('submodules')
        if isinstance(submodules, GroupedMLPSubmodules):
            grouped_mlp_submodules = submodules

        pending.append(builder.func)
        pending.extend(arg for arg in builder.args if isinstance(arg, partial))

    if not found_te_builder or grouped_mlp_submodules is None:
        raise TypeError(
            'MindSpeed alltoall overlap requires a Megatron TEGroupedMLP builder '
            f'with GroupedMLPSubmodules, but got {experts_builder!r}.'
        )
    return grouped_mlp_submodules


class AlltoAllSeqOverlapMoeLayer(BaseMoELayer):
    """
    Sets the MoE_layer when "moe-alltoall-overlap-comm" is used.
    """

    def __init__(
        self,
        config,
        submodules=None,
        layer_number=None,
        pg_collection=None,
        is_mtp_layer=False,
        name=None,
    ):
        """
        "moe-alltoall-overlap-comm" only supported "moe_grouped_gemm".
        """
        self.submodules = submodules
        self.config = config
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(config, layer_number, pg_collection, is_mtp_layer)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        from mindspeed.core.transformer.moe.moe_feature.adaptor import (
            MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor,
            MindSpeedOverLapGmmExperts,
        )

        experts_kwargs = _get_experts_builder_kwargs(self.submodules)
        # Initialize experts
        if not self.config.moe_grouped_gemm:
            raise ValueError("use '--moe-alltoall-overlap-comm' should open '--moe-grouped-gemm'.")
        else:
            self.experts = MindSpeedOverLapGmmExperts(
                self.num_local_experts,
                self.config,
                pg_collection=pg_collection,
                name=(name + ".experts") if name is not None else None,
                **experts_kwargs,
            )

        # Initialize token dispatcher
        self.token_dispatcher = MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor(
            self.num_local_experts,
            self.local_expert_indices,
            config=self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            # Use async comm linear for shared_experts.
            from mindspeed.core.transformer.moe.moe_feature.overlap.mlp_layers import (
                ShareExpertColumnParallelLinear,
                ShareExperRowParallelLinear,
            )

            # After 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed mark to 'with_shared_expert'.
            self.config.with_shared_expert = True
            shared_expert_builder = self.submodules.shared_experts
            shared_expert_submodules = shared_expert_builder.keywords['submodules']
            shared_expert_submodules.linear_fc1 = ShareExpertColumnParallelLinear
            shared_expert_submodules.linear_fc2 = ShareExperRowParallelLinear
            self.shared_experts = shared_expert_builder(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            if self.shared_expert_overlap:
                raise ValueError("use tp_extend_ep not support shared_expert_overlap.")
            self.shared_experts.with_shared_expert = True

    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        if intermediate_tensors is not None or padding_mask is not None:
            raise NotImplementedError("moe-alltoall-overlap-comm does not support intermediate_tensors or padding_mask")
        return MoELayerOverlapAllToAllSeq.apply(hidden_states, self.config, self)


class AllGatherOverlapMoeLayer(BaseMoELayer):
    '''
    Sets the MoE_layer when "moe-allgather-overlap-comm" is used.
    '''

    def __init__(
        self,
        config,
        submodules=None,
        layer_number=None,
        pg_collection=None,
        is_mtp_layer=False,
        name=None,
    ):
        """
        "moe-allgather-overlap-comm" only supported "moe_grouped_gemm".
        """
        self.submodules = submodules
        self.config = config
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(config, layer_number, pg_collection, is_mtp_layer)
        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        from mindspeed.core.transformer.moe.moe_feature.adaptor import (
            MindSpeedMOEAllGatherOverLapDispatcherAdaptor,
            MindSpeedOverLapGmmExperts,
        )

        experts_kwargs = _get_experts_builder_kwargs(self.submodules)
        # Initialize experts
        if self.config.moe_grouped_gemm:
            self.experts = MindSpeedOverLapGmmExperts(
                self.num_local_experts,
                self.config,
                pg_collection=pg_collection,
                name=(name + ".experts") if name is not None else None,
                **experts_kwargs,
            )
        else:
            raise ValueError("use '--moe-allgather-overlap-comm' should open '--moe_grouped_gemm'.")

        # Initialize token dispatcher
        self.token_dispatcher = MindSpeedMOEAllGatherOverLapDispatcherAdaptor(
            self.num_local_experts,
            self.local_expert_indices,
            config=self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            from mindspeed.core.transformer.moe.moe_feature.overlap.mlp_layers import (
                ShareExpertColumnParallelLinear,
                ShareExperRowParallelLinear,
            )

            # After 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed mark to 'with_shared_expert'.
            self.config.with_shared_expert = True
            shared_expert_builder = self.submodules.shared_experts
            shared_expert_submodules = shared_expert_builder.keywords['submodules']
            shared_expert_submodules.linear_fc1 = ShareExpertColumnParallelLinear
            shared_expert_submodules.linear_fc2 = ShareExperRowParallelLinear
            self.shared_experts = shared_expert_builder(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            if self.shared_expert_overlap:
                raise ValueError("use tp_extend_ep not support shared_expert_overlap.")
            # In 0.10.0, the definition of shared_experts has conflict. Rename the MindSpeed version to 'with_shared_expert'.
            self.shared_experts.with_shared_expert = True
        self.token_dispatcher.all_tokens_per_expert = None

    def forward(self, hidden_states):
        return MoELayerOverlapAllGather.apply(hidden_states, self.config, self)


class AlltoAllOverlapMoeLayer(MegatronBaseMoeLayer):
    """
    Sets the MoE_layer when "moe-alltoall-overlap-comm" is used.
    This function only used with 'alltoall' dispatcher.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules = None,
        layer_number: int = None,
        pg_collection=None,
        is_mtp_layer: bool = False,
        name=None,
    ):
        """
        "moe-alltoall-overlap-comm" only supported "moe_grouped_gemm".
        """

        self.submodules = submodules
        self.config = config
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.moe_layer_recompute = config.moe_layer_recompute

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        from mindspeed.core.transformer.moe.moe_feature.adaptor import (
            MindSpeedMOEAlltoAllOverLapDispatcherAdaptor,
            MindSpeedAlltoALLOverLapGmmExperts,
        )

        # Initialize experts
        if not self.config.moe_grouped_gemm:
            raise ValueError("use '--moe-alltoall-overlap-comm' should open '--moe-grouped-gemm'.")
        else:
            expert_config = copy(self.config)
            expert_config.gradient_accumulation_fusion = bool(
                getattr(self.config, 'gemm_gradient_accumulation_fusion', False)
            )
            expert_config.delay_wgrad_compute = True
            expert_config.use_transformer_engine_op_fuser = False
            grouped_mlp_submodules = _get_te_grouped_mlp_submodules(self.submodules.experts)
            self.experts = MindSpeedAlltoALLOverLapGmmExperts(
                self.num_local_experts,
                expert_config,
                submodules=grouped_mlp_submodules,
                pg_collection=pg_collection,
                name=(name + ".experts") if name is not None else None,
            )

        # Initialize token dispatcher
        self.token_dispatcher = MindSpeedMOEAlltoAllOverLapDispatcherAdaptor(
            self.num_local_experts,
            self.local_expert_indices,
            config=self.config,
            pg_collection=pg_collection,
        )

        if self.config.add_bias_linear and self.config.moe_token_dispatcher_type != 'alltoall':  # nosec B105
            self.token_dispatcher.add_bias = self.config.add_bias_linear
        else:
            self.token_dispatcher.add_bias = None

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = self.submodules.shared_experts(
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
                name=(name + ".shared_experts") if name is not None else None,
            )
            self.shared_experts.with_shared_expert = True
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        if intermediate_tensors is not None or padding_mask is not None:
            raise NotImplementedError("moe-alltoall-overlap-comm does not support intermediate_tensors or padding_mask")
        return MoELayerOverlapAllToAll.apply(hidden_states, self.config, self, None)
