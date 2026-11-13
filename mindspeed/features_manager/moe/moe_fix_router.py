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

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEFixRouterFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('fix-router', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--fix-router", action='store_true', default=False, help='Enable .')

    def validate_args(self, args):
        if args.fix_router and args.expert_model_parallel_size <= 1:
            raise AssertionError('when enable fix-router, expert_model_parallel_size must be greater than 1')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_utils import topk_softmax_with_capacity

        if args.fix_router:
            patch_manager.register_patch(
                'megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity', topk_softmax_with_capacity
            )
