from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class FusedEmaAdamwFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('optimizer-selection')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--ema-decay', type=float, default=0.9999,
                           help='Set ema_decay of fused_ema_adamw optimizer.')

    def pre_validate_args(self, args):
        self.incompatible_check(args, 'optimizer-selection')
        self.incompatible_check(args, 'ema-decay')
        self.incompatible_check(args, 'disable-gloo-group')

        if args.optimizer_selection == 'fused_ema_adamw' and args.ema_decay < 0 or args.ema_decay > 1:
            raise AssertionError("ema_decay must be in the range [0, 1].")

        if args.disable_gloo_group is None:
            args.disable_gloo_group = False

    def pre_register_patches(self, patch_manager, args):
        if args.optimization_level >= 0 and args.optimizer_selection == 'fused_ema_adamw':
            from mindspeed.core.optimizer.fused_ema_adamw.fused_ema_adamw import FusedEmaAdamW as AdamW
            patch_manager.register_patch(
                'apex.optimizers.FusedAdam', AdamW, create_dummy=True)

    def register_patches(self, patch_manager, args):
        if args.optimization_level >= 2 and args.optimizer_selection == 'fused_ema_adamw':
            from mindspeed.core.optimizer.fused_ema_adamw.adaptor import generate_state_dict_ema_wrapper, save_checkpoint_ema_wrapper
            from mindspeed.core.optimizer.fused_ema_adamw.adaptor import ema_distrib_optimizer_init_wrapper
            patch_manager.register_patch(
                'megatron.training.checkpointing.save_checkpoint', save_checkpoint_ema_wrapper)
            patch_manager.register_patch(
                'megatron.training.checkpointing.generate_state_dict', generate_state_dict_ema_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                         ema_distrib_optimizer_init_wrapper)

            if hasattr(args, "ema_decay"):
                from mindspeed.core.optimizer.fused_ema_adamw.adaptor import get_megatron_optimizer_func_wrapper
                patch_manager.register_patch('megatron.core.optimizer.get_megatron_optimizer',
                                             get_megatron_optimizer_func_wrapper)
