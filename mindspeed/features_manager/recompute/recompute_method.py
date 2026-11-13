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

from mindspeed.features_manager.feature import MindSpeedFeature


class RecomputeMethodFeature(MindSpeedFeature):
    """
    Additional setting for recompute_feature.
    """
    def __init__(self):
        super().__init__('recompute-method', optimization_level=2)

    def register_patches(self, patch_manager, args):
        self.register_basic_patches(patch_manager, args)

    def register_basic_patches(self, patch_manager, args):
        if (getattr(args, 'recompute_method', False) and args.recompute_method == 'block') and not getattr(args, 'swap_attention', False):
            from mindspeed.core.memory.common import transformer_block_checkpointed_forward
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                transformer_block_checkpointed_forward)
