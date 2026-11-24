# Copyright 2024 Huawei Technologies Co., Ltd.
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
            args.use_flash_attn = True
            LOG.warning("\"use_fusion_attn_v2\" is not recommended. This feature is not officially released.")
