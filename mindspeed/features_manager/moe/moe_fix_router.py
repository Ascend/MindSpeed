# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEFixRouterFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('fix-router', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--fix-router", action='store_true', default=False,
                           help='Enable .')

    def validate_args(self, args):
        if args.expert_model_parallel_size <= 1:
            raise argparse.ArgumentError('when enable fix-router, '
                                         'expert_model_parallel_size must be greater than 1')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_utils import topk_softmax_with_capacity
        if args.fix_router:
            patch_manager.register_patch('megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity',
                                         topk_softmax_with_capacity)