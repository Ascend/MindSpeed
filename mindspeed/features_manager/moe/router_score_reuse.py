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


class MoERouterScoreReuseFeature(MindSpeedFeature):
    """Remove duplicate router score work while preserving Megatron semantics."""

    def __init__(self):
        super().__init__("moe-router-score-reuse", 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--moe-router-score-reuse",
            action="store_true",
            default=False,
            help="Reuse identical MoE router score and Top-K results for dispatch and auxiliary loss.",
        )

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.router_score_reuse import router_score_reuse_wrapper

        patch_manager.register_patch(
            "megatron.core.transformer.moe.router.TopKRouter.routing",
            router_score_reuse_wrapper,
        )
