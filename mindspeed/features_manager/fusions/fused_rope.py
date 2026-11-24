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
import argparse
from argparse import ArgumentParser, Namespace

from mindspeed.features_manager.feature import MindSpeedFeature


class FusedRoPEFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("apply-rope-fusion", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title="fusion")

        if not self._is_arg_registered(parser, "--no-rope-fusion"):
            group.add_argument(
                "--no-rope-fusion",
                action="store_false",
                dest="apply_rope_fusion",
                default=True,
                help="Disable RoPE fusion.",
            )
        if not self._is_arg_registered(parser, "--use-fused-rotary-pos-emb"):
            group.add_argument(
                "--use-fused-rotary-pos-emb",
                action="store_true",
                dest="apply_rope_fusion",
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )

    def validate_args(self, args: Namespace):
        # Megatron owns apply_rope_fusion validation and turns it off for
        # non-RoPE position embeddings.  The deprecated alias above only
        # writes the Megatron field.
        return

    def register_patches(self, patch_manager, args: Namespace):
        from mindspeed.core.fusions.fused_rope import (
            apply_rotary_pos_emb_bshd,
            fused_apply_rotary_pos_emb_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd",
            apply_rotary_pos_emb_bshd,
        )
        patch_manager.register_patch(
            "megatron.core.extensions.transformer_engine.fused_apply_rotary_pos_emb",
            fused_apply_rotary_pos_emb_wrapper,
        )
