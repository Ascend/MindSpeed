from mindspeed.features_manager.feature import MindSpeedFeature


class ProfilerDefaultFeature(MindSpeedFeature):
    def __init__(self):
        super(ProfilerDefaultFeature, self).__init__('profiler-default', 0)

    def register_patches(self, patch_manager, args):
        from mindspeed.functional.profiler.profiler_info import get_nccl_options_add_group_info_wrapper
        patch_manager.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_add_group_info_wrapper)
