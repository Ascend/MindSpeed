# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from mindspeed.features_manager.feature import MindSpeedFeature


class HcclOpModeSetFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('hccl-op-mode')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--hccl-op-mode', type=str, default=None,
                           help='hccl op mode adaptive.')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.hccl_buffer.hccl_op_mode_adaptor import \
            get_nccl_options_wrapper, hccl_op_mode_set_wrapper
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                hccl_op_mode_set_wrapper)
