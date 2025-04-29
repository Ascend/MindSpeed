# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from logging import getLogger
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class AlibiFeature(MindSpeedFeature):
    """
    Attention positional embeding.
    To enable this feature, the reference is as follows .

    Usage:
      "--position-embedding-type alibi"
      "--alibi-fusion-attn-type [2, 3]"
      "[--alibi-diagonal-opposite]"
    """

    def __init__(self):
        super().__init__(
            'position-embedding-type',
            optimization_level=2
        )

    def is_need_apply(self, args):
        pse = getattr(args, self.feature_name, None)
        need_apply = False
        if pse == 'alibi':
            need_apply = True
        return (
                self.optimization_level <= args.optimization_level and
                need_apply
            ) or self.default_patches

    def register_args(self, parser: ArgumentParser):
        self.add_parser_argument_choices_value(
            parser,
            "--position-embedding-type",
            'alibi'
        )

        group = parser.add_argument_group(title='alibi')
        group.add_argument(
            '--square-alibi-mask',
            action='store_true',
            default=False,
            help='attention mask of alibi is squared'
        )
        group.add_argument(
            '--fill-neg-inf',
            action='store_true',
            default=False,
            help='fill alibi with negative inf'
        )

        group.add_argument(
            '--alibi-fusion-attn-type',
            type=int,
            help='alibi pse type, support for 0,2,3'
        )
        group.add_argument(
            '--alibi-diagonal-opposite',
            action='store_true',
            default=False,
            help='make alibi diagonal opposite'
        )
        
    def validate_args(self, args):
        if args.alibi_fusion_attn_type is not None:
            if args.alibi_fusion_attn_type not in [0, 2, 3]:
                raise AssertionError(
                    '--alibi-fusion-attn-type only support for `0, 2, 3`'
                )
            if args.alibi_fusion_attn_type == 0:
                raise AssertionError(
                    'fa v2 only support compress model currently. '
                    'please use 2 or 3'
                )
            # alibi is only support FA2
            if args.alibi_fusion_attn_type in [2, 3]:
                args.use_fusion_attn_v2 = True

        if args.use_fusion_attn_v2:
            args.use_flash_attn = True
            print(
                '[WARNING] \"use_fusion_attn_v2\" is not recommended.' \
                'This feature is not officially released.'
            )

    def register_patches(self, patch_manager, args):
        if int(getattr(args, 'context_parallel_size', 1)) == 1:
            from mindspeed.core.transformer.flash_attention.alibi.adaptor import MindSpeedDotProductAttention
            patch_manager.register_patch(
                'megatron.core.transformer.dot_product_attention.DotProductAttention',
                MindSpeedDotProductAttention
        )