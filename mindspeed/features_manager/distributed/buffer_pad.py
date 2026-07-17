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
        if getattr(args, self.feature_name, None) is not None:
            raise NotImplementedError(
                '--param-and-grad-buffer-pad is not supported by the scoped Megatron 0.18 adaptation.'
            )

    def register_patches(self, patch_manager, args):
        return
