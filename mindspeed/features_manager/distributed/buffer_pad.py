# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class BufferPadFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('param-and-grad-buffer-pad', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--param-and-grad-buffer-pad', type=int, default=None,
                           help='Use this argument to ensure that all buckets start at a memory address that is needed-byte. Set 512 for Ascend')

    def validate_args(self, args):
        if args.param_and_grad_buffer_pad and args.param_and_grad_buffer_pad <= 0:
            raise AssertionError('--param-and-grad-buffer-pad must be greater than 0')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.buffer_pad.adaptor import param_and_grad_buffer_init_pad
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
                                param_and_grad_buffer_init_pad)



