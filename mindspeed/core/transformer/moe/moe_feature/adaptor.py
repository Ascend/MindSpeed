# Copyright (c) 2025, Huawei Technologies.
# All rights reserved.

from mindspeed.core.transformer.moe.moe_feature import MoELayer as MegatronMoELayer
from mindspeed.core.transformer.moe.moe_feature import GroupedMLP as MegatronGroupedMLP
from mindspeed.core.transformer.moe.moe_feature import MoEAlltoAllSEQTokenDispatcher as MegatronMoEAlltoAllSEQTokenDispatcher
from mindspeed.core.transformer.moe.moe_feature import MoEAllGatherTokenDispatcher as MegatronMoEAllGatherTokenDispatcher

from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.moe_layer import All2AllSeqTpExtendEpMoELayerImpl
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import All2AllSeqTp2epDispatcherImpl
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.experts import TpExtendEpGmmExpertsImpl

from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer import AlltoAllSeqOverlapMoeLayer, AllGatherOverlapMoeLayer
from mindspeed.core.transformer.moe.moe_feature.overlap.token_dispatcher import MoEAlltoAllSeqOverLapDispatcher, MoEAllGatherOverLapDispatcher
from mindspeed.core.transformer.moe.moe_feature.overlap.experts import OverLapGmmExpertsImpl

from mindspeed.core.transformer.moe.moe_feature.gmm.experts import GmmExpertsImpl


class MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher(All2AllSeqTp2epDispatcherImpl, MegatronMoEAlltoAllSEQTokenDispatcher):
    # TokenDispatcher of AlltoAllSEQ API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        All2AllSeqTp2epDispatcherImpl.__init__(self, *args, **kwargs)


class MindSpeedTpExtendEpGmmExperts(TpExtendEpGmmExpertsImpl, MegatronGroupedMLP):
    # GroupedGEMM API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        TpExtendEpGmmExpertsImpl.__init__(self, *args, **kwargs)


class MindSpeedAlltoAllSEQTptoEpMoELayer(All2AllSeqTpExtendEpMoELayerImpl, MegatronMoELayer):
    # MoELayer of AlltoAllSEQ API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        if not hasattr(kwargs['config'], 'shared_expert_gate'):
            kwargs['config'].shared_expert_gate = None

        # shared_expert two param mutual conversion
        if kwargs['config'].n_shared_experts:
            kwargs['config'].moe_shared_expert_intermediate_size = kwargs['config'].n_shared_experts * \
                                                                   kwargs['config'].moe_ffn_hidden_size
        All2AllSeqTpExtendEpMoELayerImpl.__init__(self, *args, **kwargs)
        
        
class MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor(AlltoAllSeqOverlapMoeLayer, MegatronMoELayer):
    # MoELayer of AlltoAllSEQ overlap API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        if not hasattr(kwargs['config'], 'shared_expert_gate'):
            kwargs['config'].shared_expert_gate = None

        AlltoAllSeqOverlapMoeLayer.__init__(self, *args, **kwargs)


class MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor(MoEAlltoAllSeqOverLapDispatcher, MegatronMoEAlltoAllSEQTokenDispatcher):
    # TokenDispatcher of AlltoAllSEQ overlap API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        MoEAlltoAllSeqOverLapDispatcher.__init__(self, *args, **kwargs)


class MindSpeedAllGatherOverlapMoeLayerAdaptor(AllGatherOverlapMoeLayer, MegatronMoELayer):
    # MoELayer of AllGather with overlap.
    def __init__(self, *args, **kwargs):
        if not hasattr(kwargs['config'], 'shared_expert_gate'):
            kwargs['config'].shared_expert_gate = None

        AllGatherOverlapMoeLayer.__init__(self, *args, **kwargs)


class MindSpeedMOEAllGatherOverLapDispatcherAdaptor(MoEAllGatherOverLapDispatcher, MegatronMoEAllGatherTokenDispatcher):
    # TokenDispatcher of AllGather API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        MoEAllGatherOverLapDispatcher.__init__(self, *args, **kwargs)


class MindSpeedOverLapGmmExperts(OverLapGmmExpertsImpl, MegatronGroupedMLP):
    # GroupedGEMM with overlap API
    def __init__(self, *args, **kwargs):
        OverLapGmmExpertsImpl.__init__(self, *args, **kwargs)


class MindSpeedGmmExperts(GmmExpertsImpl, MegatronGroupedMLP):
    # GroupedGEMM API
    def __init__(self, *args, **kwargs):
        GmmExpertsImpl.__init__(self, *args, **kwargs)
