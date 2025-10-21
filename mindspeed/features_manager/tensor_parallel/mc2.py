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
            if getattr(args, 'use_ascend_coc', None):
                raise AssertionError('mc2 and coc can not be used together')
            if hasattr(args, 'sequence_parallel') and not args.sequence_parallel or args.tensor_model_parallel_size == 1:
                warnings.warn("The 'mc2' feature requires both the Tensor Model Parallel (TP) size to be greater than 1 "
                              "and Sequence Parallelism to be enabled. Currently, either the TP size is set to 1 or "
                              "Sequence Parallelism is disabled. As a consequence, the 'mc2' feature has been "
                              "automatically disabled. Please ensure that you configure the TP size to be greater than "
                              "1 and enable Sequence Parallelism to utilize the 'mc2' feature.")
            if getattr(args, 'use_pipe_experts', None):
                raise AssertionError('mc2 is not compatible with use_pipe_experts')
            if getattr(args, 'use_nanopipe', None):
                raise AssertionError('mc2 is not compatible with use_nanopipe')

    def register_patches(self, patch_manager, args):
        if hasattr(args, 'sequence_parallel') and not args.sequence_parallel or int(args.tensor_model_parallel_size) == 1:
            return
        from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2ColumnParallelLinear
        from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2RowParallelLinear
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                    MindSpeedMC2ColumnParallelLinear)
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                    MindSpeedMC2RowParallelLinear)
        # TE linear mc2 patch.
        patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelLinear',
                                    MindSpeedMC2ColumnParallelLinear)
        patch_manager.register_patch('megatron.core.extensions.transformer_engine.TERowParallelLinear',
                                    MindSpeedMC2RowParallelLinear)
