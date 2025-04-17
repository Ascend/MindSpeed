from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class TflopsCalculateFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('op-cal-tflops', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--op-cal-tflops', action='store_true', default=False,
                           help='use for cal mfu and hfu')

    def register_patches(self, patch_manager, args):
        from mindspeed.functional.tflops_calculate.adaptor import training_log
        from mindspeed.functional.tflops_calculate.tflops_utils import checkpoint_function_backward_wrapper
        from mindspeed.functional.tflops_calculate.tflops_utils import train_step_wrapper
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.training.training.training_log', training_log)
            patch_manager.register_patch('mindspeed.core.tensor_parallel.random.checkpoint_function_backward', checkpoint_function_backward_wrapper)
            patch_manager.register_patch('megatron.training.training.train_step', train_step_wrapper)





