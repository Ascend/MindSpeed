from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ReuseFP32Param(MindSpeedFeature):

    def __init__(self):
        super().__init__('reuse-fp32-param', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--reuse-fp32-param', action='store_true',
                           help='The distributed training optimizer frees up '
                           'param copies of FP32 to save memory.')

    def validate_args(self, args):
        self.incompatible_check(args, 'enable_zero3')
        self.dependency_check(args, 'bf16')

        if args.reuse_fp32_param and args.use_legacy_models and args.overlap_param_gather:
            raise AssertionError('In legacy, `overlap_param_gather` does not support `reuse_fp32_param`.')
        if args.reuse_fp32_param and args.optimizer_selection == 'fused_ema_adamw':
            raise AssertionError('fused_ema_adamw optimizer is not compatible with reuse_fp32_param')

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.memory.reuse_param.adaptor import reuse_fp32_param_distrib_optimizer_init_wrapper
            from mindspeed.core.memory.reuse_param.adaptor import (step_with_ready_grads, prepare_grads,
                                            reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper)
            from mindspeed.core.memory.reuse_param.adaptor import reuse_fp32_param_param_and_grad_buffer_init_wrapper

            # optim relative.
            patch_manager.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
                                         prepare_grads)
            patch_manager.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step_with_ready_grads',
                                          step_with_ready_grads)
            patch_manager.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                                         reuse_fp32_param_init_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                                         optimizer_config_init_wrapper)

            if not getattr(args, 'enable_zero3', False) and args.optimizer_selection != 'fused_ema_adamw':
                patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                             reuse_fp32_param_distrib_optimizer_init_wrapper)
            if not getattr(args, 'param_and_grad_buffer_pad', None):
                patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
                                             reuse_fp32_param_param_and_grad_buffer_init_wrapper)