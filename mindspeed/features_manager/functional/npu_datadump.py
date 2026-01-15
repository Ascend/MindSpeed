from mindspeed.features_manager.feature import MindSpeedFeature


class NPUDataDumpFeature(MindSpeedFeature):
    def __init__(self):
        super(NPUDataDumpFeature, self).__init__("npu-datadump")

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--npu-datadump', action='store_true', default=False,
                           help='enable npu data dump with mstt.')

    def register_patches(self, patch_manager, args):
        if args.npu_datadump:
            try:
                from msprobe.pytorch import PrecisionDebugger
            except ImportError as e:
                raise AssertionError('Mstt not found. You can install it with `pip install mindstudio-probe`.') from e

            from mindspeed.functional.npu_datadump.npu_datadump import dump_start_wrapper, dump_end_wrapper
            patch_manager.register_patch('megatron.training.ft_integration.on_training_step_start', dump_start_wrapper)
            patch_manager.register_patch('megatron.training.ft_integration.on_training_step_end', dump_end_wrapper)
