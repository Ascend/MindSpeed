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


class ProfilerDefaultFeature(MindSpeedFeature):
    def __init__(self):
        super(ProfilerDefaultFeature, self).__init__('profiler-default', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed.functional.profiler.profiler_info import get_nccl_options_add_group_info_wrapper
        patch_manager.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_add_group_info_wrapper)
