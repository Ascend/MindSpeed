# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

from mindspeed.features_manager.feature import MindSpeedFeature


class GenerateMaskFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__(
            'no-create-attention-mask-in-dataloader',
            optimization_level=2
        )

    def is_need_apply(self, args: Any) -> bool:
        """Check the feature is need to apply."""
        need_apply = True
        
        # can't find feature name, need to enable
        if getattr(args, self.feature_name, None):
            need_apply = False

        return (
                self.optimization_level <= args.optimization_level and
                need_apply
            ) or self.default_patches
    
    def validate_args(self, args):
        args.create_attention_mask_in_dataloader = False

    def register_patches(self, patch_manager, args):
        _cp_algo = getattr(args, 'context_parallel_algo', 'megatron_cp_algo')
        _cp_expanded_by_2d_tp = getattr(args, 'tp_2d', False) and getattr(args, 'tp_y', 1) > 1
        if int(getattr(args, 'context_parallel_size', 1)) < 2 and not (_cp_expanded_by_2d_tp and _cp_algo == 'megatron_cp_algo'):
            from mindspeed.core.transformer.flash_attention.generate_mask.adaptor import dot_product_attention_forward_wrapper
            patch_manager.register_patch(
                'megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                dot_product_attention_forward_wrapper
            )
