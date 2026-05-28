from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class SwapOptimizerFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('swap-optimizer')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--swap-optimizer', action='store_true', help='swap optimizer to cpu')
        group.add_argument(
            '--swap-optimizer-times',
            type=int,
            default=16,
            help='Each swap will be moved (len(shard_fp32_from_float16) // swap_optimizer_times) elements',
        )

    def validate_args(self, args):
        self.incompatible_check(args, 'reuse_fp32_param')
        if (
            getattr(args, self.feature_name, None)
            and not getattr(args, "use_distributed_optimizer", None)
            and not getattr(args, "use_layer_wise_distributed_optimizer", None)
        ):
            raise ValueError("Only support swap-optimizer when using distributed optimizer")

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            if 'adam' in getattr(args, 'optimizer', 'adam'):
                from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import (
                    SwapDistributedOptimizer,
                    swap_adamw_step,
                )

                patch_manager.register_patch(
                    'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer', SwapDistributedOptimizer
                )
                patch_manager.register_patch('mindspeed.core.optimizer.adamw.AdamW.step', swap_adamw_step)
            elif 'muon' in getattr(args, 'optimizer', 'adam'):
                from mindspeed.core.optimizer.swap_muon.swap_muon import (
                    swap_layer_wise_distributed_optimizer_init_wrapper,
                    swap_muon_step,
                )

                patch_manager.register_patch(
                    'mindspeed.core.optimizer.muon.layer_wise_optimizer.LayerWiseDistributedOptimizer.__init__',
                    swap_layer_wise_distributed_optimizer_init_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.optimizer.muon.emerging_optimizers.TensorParallelMuon.step', swap_muon_step
                )
