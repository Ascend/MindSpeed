# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class HcclBufferFAdaptiveFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('hccl-group-buffer-adaptive')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--hccl-group-buffer-adaptive', action='store_true', default=False,
                            help='the hccl buffer for group adaptively')
        group.add_argument('--hccl-ep-group-buffer-adaptive-factor', type=float, default=-1.0,
                            help='the ep group buffer factor')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.hccl_buffer.adaptor import \
            get_nccl_options_wrapper, hccl_buffer_adaptive_wrapper
        from mindspeed.arguments import core_transformer_config_from_args_wrapper
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                hccl_buffer_adaptive_wrapper)
            patch_manager.register_patch('megatron.training.arguments.core_transformer_config_from_args',
                        core_transformer_config_from_args_wrapper)
