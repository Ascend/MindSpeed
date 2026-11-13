# Copyright 2026 Huawei Technologies Co., Ltd.
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
