# Copyright (c) 2025, Huawei Technologies.
# All rights reserved.

from mindspeed.core.transformer.moe.moe_feature import MoELayer as MegatronMoELayer
from mindspeed.core.transformer.moe.moe_feature import GroupedMLP as MegatronGroupedMLP
from mindspeed.core.transformer.moe.moe_feature import MoEAlltoAllSEQTokenDispatcher as MegatronMoEAlltoAllSEQTokenDispatcher
from mindspeed.core.transformer.moe.moe_feature import MoEAllGatherTokenDispatcher as MegatronMoEAllGatherTokenDispatcher
from mindspeed.core.transformer.moe.moe_feature import MoEAlltoAllTokenDispatcher as MegatronMoEAlltoAllTokenDispatcher

from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.moe_layer import All2AllSeqTpExtendEpMoELayerImpl
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import All2AllSeqTp2epDispatcherImpl
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.experts import TpExtendEpGmmExpertsImpl

from mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer import (AlltoAllSeqOverlapMoeLayer, 
                                                                          AllGatherOverlapMoeLayer,
                                                                          AlltoAllOverlapMoeLayer
                                                                         )
from mindspeed.core.transformer.moe.moe_feature.overlap.token_dispatcher import (MoEAlltoAllSeqOverLapDispatcher,
                                                                                 MoEAllGatherOverLapDispatcher,
                                                                                 MoEAlltoAllOverLapDispatcher
                                                                                )
from mindspeed.core.transformer.moe.moe_feature.overlap.experts import OverLapGmmExpertsImpl, AlltoAllOverLapGmmExpertsImpl
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

        # shared_expert two param mutual conversion
        if kwargs['config'].n_shared_experts:
            kwargs['config'].moe_shared_expert_intermediate_size = (
                kwargs['config'].n_shared_experts *
                (kwargs['config'].moe_ffn_hidden_size
                 if kwargs['config'].moe_ffn_hidden_size is not None
                 else kwargs['config'].ffn_hidden_size)
            )
        All2AllSeqTpExtendEpMoELayerImpl.__init__(self, *args, **kwargs)
        
        
class MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor(AlltoAllSeqOverlapMoeLayer, MegatronMoELayer):
    # MoELayer of AlltoAllSEQ overlap API which support tp_extend_ep
    def __init__(self, *args, **kwargs):

        AlltoAllSeqOverlapMoeLayer.__init__(self, *args, **kwargs)


class MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor(MoEAlltoAllSeqOverLapDispatcher, MegatronMoEAlltoAllSEQTokenDispatcher):
    # TokenDispatcher of AlltoAllSEQ overlap API which support tp_extend_ep
    def __init__(self, *args, **kwargs):
        MoEAlltoAllSeqOverLapDispatcher.__init__(self, *args, **kwargs)


class MindSpeedAllGatherOverlapMoeLayerAdaptor(AllGatherOverlapMoeLayer, MegatronMoELayer):
    # MoELayer of AllGather with overlap.
    def __init__(self, *args, **kwargs):

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


#------------for alltoall dispatcher
class MindSpeedAlltoAllOverlapMoeLayerAdaptor(AlltoAllOverlapMoeLayer):
    # MoELayer of AlltoAll overlap API.
    def __init__(self, *args, **kwargs):
        AlltoAllOverlapMoeLayer.__init__(self, *args, **kwargs)


class MindSpeedMOEAlltoAllOverLapDispatcherAdaptor(MoEAlltoAllOverLapDispatcher, MegatronMoEAlltoAllTokenDispatcher):
    # TokenDispatcher of AlltoAll overlap API.
    def __init__(self, *args, **kwargs):
        MoEAlltoAllOverLapDispatcher.__init__(self, *args, **kwargs)


class MindSpeedAlltoALLOverLapGmmExperts(AlltoAllOverLapGmmExpertsImpl, MegatronGroupedMLP):
    # GroupedGEMM API
    def __init__(self, *args, **kwargs):
        AlltoAllOverLapGmmExpertsImpl.__init__(self, *args, **kwargs)
