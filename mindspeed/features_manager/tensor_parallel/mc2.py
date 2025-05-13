# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import warnings
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MC2Feature(MindSpeedFeature):

    def __init__(self):
        super().__init__('use-ascend-mc2', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-ascend-mc2", action='store_true',
                           help="Use ascend mc2")

    def validate_args(self, args):
        self.incompatible_check(args, 'unaligned_linear')
        if args.use_ascend_mc2:
            if args.num_experts:
                raise AssertionError('mc2 is not compatible with moe models')
            if getattr(args, 'use_ascend_coc', None):
                raise AssertionError('mc2 and coc can not be used together')
            if not args.sequence_parallel or args.tensor_model_parallel_size == 1:
                warnings.warn("The use_ascend_mc2 feature requires tp to be greater than 1 and sequence_parallel is "
                              "enabled. However, currently tp is equal to 1 or sequence parallelism is not enabled, "
                              "and use_ascend_mc2 feature has been disabled.")
            if getattr(args, 'use_pipe_experts', None):
                raise AssertionError('mc2 is not compatible with use_pipe_experts')
            if getattr(args, 'use_nanopipe', None):
                raise AssertionError('mc2 is not compatible with use_nanopipe')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2ColumnParallelLinear
        from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2RowParallelLinear
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                     MindSpeedMC2ColumnParallelLinear)
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                     MindSpeedMC2RowParallelLinear)
