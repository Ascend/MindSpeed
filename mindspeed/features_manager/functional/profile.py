# Copyright 2025 Huawei Technologies Co., Ltd.
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
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ProfileFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('profile', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--profile-level", type=str, default='level0',
                           choices=['level0', 'level1', 'level2'],
                           help="Profile level default level0.")
        group.add_argument("--profile-with-cpu", action='store_true', default=False,
                           help="Profile with cpu info.")
        group.add_argument("--profile-with-stack", action='store_true', default=False,
                           help="Profile with stack info.")
        group.add_argument("--profile-with-memory", action='store_true', default=False,
                           help="Profile with memory info.")
        group.add_argument("--profile-record-shapes", action='store_true', default=False,
                           help="Profile record shape info.")
        group.add_argument("--profile-save-path", type=str, default='./profile_dir',
                           help="Profile save path.")

    def register_patches(self, patch_manager, args):
        from mindspeed.functional.profile.adaptor import train_wrapper, train_step_wrapper
        patch_manager.register_patch('megatron.training.training.train', train_wrapper)
        patch_manager.register_patch('megatron.training.training.train_step', train_step_wrapper)
