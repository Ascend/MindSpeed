# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class BufferPadFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('param-and-grad-buffer-pad', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--param-and-grad-buffer-pad',
            type=int,
            default=None,
            help='Use this argument to ensure that all buckets start at a memory address that is needed-byte. Set 512 for Ascend',
        )

    def validate_args(self, args):
        alignment = getattr(args, self.feature_name, None)
        if alignment is not None and alignment <= 0:
            raise AssertionError('--param-and-grad-buffer-pad must be greater than 0')
        if alignment and getattr(args, 'use_layer_wise_distributed_optimizer', False):
            raise NotImplementedError(
                '--param-and-grad-buffer-pad does not support the layer-wise distributed optimizer.'
            )

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.distributed.buffer_pad.adaptor import compute_aligned_per_buffer_param_layout

            patch_manager.register_patch(
                'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._compute_per_buffer_param_layout',
                compute_aligned_per_buffer_param_layout,
            )
