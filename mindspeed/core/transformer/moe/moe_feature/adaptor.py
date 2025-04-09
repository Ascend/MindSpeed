# Copyright (c) 2025, Huawei Technologies.
# All rights reserved.
from mindspeed.core.transformer.moe.moe_feature import MoELayer as MegatronMoELayer
from mindspeed.core.transformer.moe.moe_feature import GroupedMLP as MegatronGroupedMLP
from mindspeed.core.transformer.moe.moe_feature import MoEAlltoAllSEQTokenDispatcher as MegatronMoEAlltoAllSEQTokenDispatcher

from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.moe_layer import All2AllSeqTpExtendEpMoELayerImpl
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import All2AllSeqTp2epDispatcherImpl
from mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.experts import TpExtendEpGmmExpertsImpl

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
                                                                   kwargs['config'].ffn_hidden_size
        All2AllSeqTpExtendEpMoELayerImpl.__init__(self, *args, **kwargs)
        
        
class MindSpeedGmmExperts(GmmExpertsImpl, MegatronGroupedMLP):
    # GroupedGEMM API
    def __init__(self, *args, **kwargs):
        GmmExpertsImpl.__init__(self, *args, **kwargs)
