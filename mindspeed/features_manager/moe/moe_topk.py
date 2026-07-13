# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEExpertCapacityFactorFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('moe-expert-capacity-factor', 0)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_utils import apply_router_token_dropping

        if getattr(args, 'moe_expert_capacity_factor', None):
            patch_manager.register_patch(
                'megatron.core.transformer.moe.moe_utils.apply_router_token_dropping', apply_router_token_dropping
            )
