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

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class QosFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('aiqos', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--aiqos', action='store_true', help='use ai qos feature')
        group.add_argument('--aiqos-mode', type=str, default='auto', help='ai qos mode')
        group.add_argument('--aiqos-schedule', type=str, help='ai qos schedule')
        group.add_argument('--aiqos-enable-roce', action='store_true', help='ai qos roce enable')

    def is_need_apply(self, args):
        return self.optimization_level <= args.optimization_level

    def register_patches(self, patch_manager, args):
        """
        Register QoS patches and validate manual mode parameters

        Args:
            patch_manager: Patch manager instance for registering patches
            args: Parsed command line arguments
        """
        if not args.aiqos:
            return

        if args.aiqos_mode == 'manual' and not args.aiqos_schedule:
            raise ValueError("QoS manual mode requires --aiqos-schedule parameter. ")

        from mindspeed.core.qos.adaptor import get_nccl_options_qos_wrapper

        # Patch get_nccl_options so both create_group and direct new_group paths get QoS.
        patch_manager.register_patch('megatron.core.parallel_state.get_nccl_options', get_nccl_options_qos_wrapper)
