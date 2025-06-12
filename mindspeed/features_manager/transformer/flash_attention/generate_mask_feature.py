# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any
from argparse import ArgumentParser

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
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='generate mask')
        group.add_argument(
            '--sparse-mode',
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            help=(
                'To improve performance in different modes of attention mask '
                '0: defaultMask '
                '1: allMask '
                '2: leftUpCausal '
                '3: rightDownCausal '
                '4: band '
                '5: prefix compressed '
                '6: prefix no uncompressed '
                '7: varlen - rightDownCausal '
                '8: varlen - leftUpCausal '
            )
        )

    def register_patches(self, patch_manager, args):
        if int(getattr(args, 'context_parallel_size', 1)) < 2:
            from mindspeed.core.transformer.flash_attention.generate_mask.adaptor import dot_product_attention_forward_wrapper
            patch_manager.register_patch(
                'megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                dot_product_attention_forward_wrapper
            )
