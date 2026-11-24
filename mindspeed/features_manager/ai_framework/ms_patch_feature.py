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


class MindSporePatchFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('mindspore-patch', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--ai-framework', type=str, default='pytorch', help='support pytorch and mindspore')

    def register_patches(self, patch_manager, args):
        if not hasattr(args, "ai_framework") or args.ai_framework != "mindspore" or args.optimization_level < 0:
            return
        from mindspeed.mindspore.mindspore_adaptor import mindspore_adaptation
        mindspore_adaptation(patch_manager, args)
