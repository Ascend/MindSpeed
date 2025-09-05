# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class CoCFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('use-ascend-coc', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-ascend-coc", action='store_true',
                           help="Use ascend coc")
        group.add_argument('--coc-mode', type=int, default=-1,
                           help='coc-mode: 0=original, 1=rewrite, 2=coc default')
        group.add_argument('--coc-parallel-num', type=int, default=1,
                           help='coc parallel num')
        group.add_argument('--coc-fused-kernel', action='store_true',
                           help='use coc fused kernel')

    def validate_args(self, args):
        self.incompatible_check(args, 'unaligned_linear')
        if hasattr(args, 'use_ascend_mc2') and args.use_ascend_mc2:
            if args.use_ascend_coc:
                raise AssertionError('--mc2 and coc can not be used together')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCColumnParallelLinear
        from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCRowParallelLinear
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                     MindSpeedCoCColumnParallelLinear)
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                     MindSpeedCoCRowParallelLinear)
