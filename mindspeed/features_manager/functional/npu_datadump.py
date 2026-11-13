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


class NPUDataDumpFeature(MindSpeedFeature):
    def __init__(self):
        super(NPUDataDumpFeature, self).__init__("npu-datadump")

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--npu-datadump', action='store_true', default=False,
                           help='enable npu data dump with mstt.')

    def register_patches(self, patch_manager, args):
        if args.npu_datadump:
            try:
                from msprobe.pytorch import PrecisionDebugger
            except ImportError as e:
                raise AssertionError('Mstt not found. You can install it with `pip install mindstudio-probe`.') from e

            from mindspeed.functional.npu_datadump.npu_datadump import dump_start_wrapper, dump_end_wrapper
            patch_manager.register_patch('megatron.training.training.train_step', dump_start_wrapper)
            patch_manager.register_patch('megatron.training.ft_integration.on_training_step_end', dump_end_wrapper)
