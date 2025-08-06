# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class MindSporePatchFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('mindspore-patch', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--ai-framework', type=str, default='pytorch', help='support pytorch and mindspore')

    def register_patches(self, patch_manager, args):
        if not hasattr(args, "ai_framework") or args.ai_framework != "mindspore" or args.optimization_level < 0:
            return
        from mindspeed.mindspore.mindspore_adaptor import mindspore_adaptation
        mindspore_adaptation(patch_manager, args)
