# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from mindspeed.features_manager.feature import MindSpeedFeature


class MegatronMcoreRearrangeFeature(MindSpeedFeature):
    """
    rearrange attention's init&forward for overlap_param_gather.
    """
    def __init__(self):
        super().__init__('mcore-rearrange', optimization_level=0)

    def register_patches(self, patch_manager, args):
        self.register_basic_patches(patch_manager, args)

    def register_basic_patches(self, pm, args):
        # args parser patch
        from mindspeed.features_manager.transformer.module_rearrange.mcore_model_rearrange import (mindspeed_self_attention_init_wrapper, \
                                                                                                    megatron_module_init_wrapper
                                                                                                    )
        pm.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', mindspeed_self_attention_init_wrapper)
        pm.register_patch('megatron.core.transformer.module.MegatronModule.__init__', megatron_module_init_wrapper)
