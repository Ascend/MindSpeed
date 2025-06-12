# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from argparse import ArgumentParser, Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature

LOG = getLogger(__name__)


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

    def validate_args(self, args: Namespace):
        if args.use_fusion_attn_v2:
            if args.use_flash_attn:
                raise AssertionError(
                    'can not enable fav1 and fav2 simultaneously'
                )
