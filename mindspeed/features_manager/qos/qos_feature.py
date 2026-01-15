from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class QosFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('aiqos', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--aiqos', action='store_true', help='use ai qos feature')
        group.add_argument('--aiqos-mode', type=str, default='auto', help='ai qos mode')
        group.add_argument('--aiqos-schedule', type=str, help='ai qos schedule')

    def is_need_apply(self, args):
        return self.optimization_level <= args.optimization_level

    def register_patches(self, patch_manager, args):
        """
        Register QoS patches and validate manual mode parameters

        Args:
            patch_manager: Patch manager instance for registering patches
            args: Parsed command line arguments
        """
        if args.aiqos:
            # Validate manual mode requirements
            if args.aiqos_mode == 'manual':
                # Check if schedule is provided
                if not args.aiqos_schedule:
                    raise ValueError(
                        "QoS manual mode requires --qos-schedule parameter. "
                    )
            # Import QoS modules and register patches
            from mindspeed.core.qos.adaptor import create_group_qos, initialize_model_parallel_qos
            patch_manager.register_patch(
                'megatron.core.parallel_state.initialize_model_parallel',
                initialize_model_parallel_qos
            )
