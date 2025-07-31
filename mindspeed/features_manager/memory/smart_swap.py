import time
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature

SWAP_DELAY = 10


class SmartSwapFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('smart-swap', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--smart-swap',
                           action='store_true', default=False, help='Enable the smart swap feature.')

    def validate_args(self, args):
        adaptive_recompute_enable = getattr(args, "adaptive_recompute_device_size", -1) > 0 or getattr(args, "adaptive-recompute-device-swap", False)
        if args.smart_swap and adaptive_recompute_enable:
            raise AssertionError('smart swap is not compatible with adaptive selective recompute')

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            adaptive_recompute_enable = getattr(args, "adaptive_recompute_device_size", -1) > 0 or getattr(args, "adaptive-recompute-device-swap", False)
            if not adaptive_recompute_enable:
                from mindspeed.core.memory.smart_swap.swap_adaptor import change_allocator
                time.sleep(SWAP_DELAY)
                change_allocator()
                from mindspeed.core.memory.smart_swap.swap_megatron_adaptor import train_step_wrapper
                patch_manager.register_patch('megatron.training.training.train_step', train_step_wrapper)