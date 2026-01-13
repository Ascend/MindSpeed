# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

import torch

from mindspeed.features_manager.feature import MindSpeedFeature


class CompressActivationFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('compress-activation', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--compress-activation', type=str, default='',
                           help='Compress activation in each layer.')

    def validate_args(self, args):
        if args.compress_activation != "":
            if args.compress_activation == '0':
                args.compress_activation = list(range(1, args.num_layers + 1))
            else:
                compress_layers = []
                layer_args = args.compress_activation.split(',')
                for elem in layer_args:
                    if '-' in elem:
                        start, end = map(int, elem.split('-'))
                        compress_layers.extend(range(start, end + 1))
                    else:
                        compress_layers.append(int(elem))
                args.compress_activation = compress_layers

    def register_patches(self, patch_manager, args):
        from mindspeed.core.memory.compress.adaptor import layer_forward_wrapper
        if getattr(args, "compress_activation", False):
            patch_manager.register_patch('megatron.core.transformer.TransformerLayer.forward', layer_forward_wrapper)


class CompressOptimizerFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('compress-optimizer', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--compress-optimizer', action='store_true', default=False,
                       help='Compress optimizer states.')

    def validate_args(self, args):
        if args.compress_optimizer != "disable":
            import torch_npu
            if not hasattr(torch_npu, "npu_hans_encode") or not hasattr(torch_npu, "npu_hans_decode") \
                or not hasattr(torch_npu, "empty_with_swapped_memory"):
                raise AssertionError("`--compress-optimizer` is invalid, please update the latest PTA version.")
            self.incompatible_check(args, "fused_ema_adamw")

    def register_patches(self, patch_manager, args):
        from mindspeed.core.memory.compress.adaptor import compress_optimizer_step
        if getattr(args, "compress_optimizer", False):
            patch_manager.register_patch('mindspeed.optimizer.adamw.AdamW.step', compress_optimizer_step)
            patch_manager.register_patch('mindspeed.core.optimizer.adamw.AdamW.step', compress_optimizer_step)
