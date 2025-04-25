# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class HcclBufferSetFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('hccl-group-buffer')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--hccl-group-buffer', type=str, default=None, help='the hccl buffer for group')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.hccl_buffer.adaptor import \
            get_nccl_options_wrapper, hccl_buffer_set_wrapper

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                hccl_buffer_set_wrapper)

