# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class FusionAttentionFeature(MindSpeedFeature):
    '''
    fusion attention.

    '''

    def __init__(self):
        super().__init__(
            'use-flash-attn',
            optimization_level=2
        )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='fusion attention')
        group.add_argument(
            '--pre-tockens',
            type=int,
            default=65536,
            help='pre-tockens is used by Flash attention'
        )
        group.add_argument(
            '--next-tockens',
            type=int,
            default=0,
            help='next-tockens is used by Flash attention'
        )
        group.add_argument(
            '--sparse-mode',
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            help=(
                'mask type for fusion attention '
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

    def validate_args(self, args):
        if args.use_flash_attn and not (args.sparse_mode == 0 or args.sparse_mode == 2):
            raise AssertionError("When use_flash_attn, only supports sparse modes 0 and 2")

    def register_patches(self, patch_manager, args):
        if int(getattr(args, 'context_parallel_size', 1)) < 2:
            from mindspeed.core.transformer.flash_attention.flash_attention.adaptor import \
                dot_product_attention_forward_impl

            patch_manager.register_patch(
                'megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                dot_product_attention_forward_impl
            )
