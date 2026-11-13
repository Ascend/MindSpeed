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

from logging import getLogger
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class AlibiFeature(MindSpeedFeature):
    """
    Attention positional embedding.
    To enable this feature, the reference is as follows .

    Usage:
      "--position-embedding-type alibi"
      "--alibi-fusion-attn-type 0 or 2"
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
            help='alibi pse type, support for 0,2'
        )
        group.add_argument(
            '--alibi-diagonal-opposite',
            action='store_true',
            default=False,
            help='make alibi diagonal opposite'
        )
        
    def validate_args(self, args):
        if args.alibi_fusion_attn_type is not None:
            if args.alibi_fusion_attn_type not in [0, 2]:
                raise AssertionError(
                    '--alibi-fusion-attn-type only support for `0, 2`'
                )
            # alibi is only support FA2
            if args.alibi_fusion_attn_type in [0, 2]:
                args.use_fusion_attn_v2 = True

    def register_patches(self, patch_manager, args):
        if int(getattr(args, 'context_parallel_size', 1)) == 1:
            from mindspeed.core.transformer.flash_attention.alibi.adaptor import MindSpeedDotProductAttention
            patch_manager.register_patch(
                'megatron.core.transformer.dot_product_attention.DotProductAttention',
                MindSpeedDotProductAttention
        )