# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class EnableRecomputeLayersPerPPRank(MindSpeedFeature):
    def __init__(self):
        super().__init__('enable-recompute-layers-per-pp-rank')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-recompute-layers-per-pp-rank',
                           action='store_true', default=False,
                           help='If enabled, --recompute-num-layers will mean the number of '
                           'layers recomputed in each pp rank. Otherwise it means the number '
                           'of layers recomputed in each vpp rank.')
