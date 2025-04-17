# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class FusionAttentionV2Feature(MindSpeedFeature):
    '''
    fusion attention v2 is a expand to fusion attention v1
    and only support for alibi positional embeding currently.
    Close by default.
    '''

    def __init__(self):
        super().__init__(
            'use-fusion-attn-v2', 
            optimization_level=2
        )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='fusion attention v2')
        group.add_argument(
            '--use-fusion-attn-v2',
            action='store_true',
            default=False,
            help='enalbe fusion attention v2'
        )
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
            help='mask type for fusion attention'
        )
        
    def validate_args(self, args):
        if args.use_fusion_attn_v2:
            args.use_flash_attn = True
