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


class MoESharedExpertsFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('n-shared-experts', 0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--n-shared-experts', type=int, default=None,
                           help='DEPRECATED. use --moe-shared-expert-intermediate-size replace')

    def validate_args(self, args):
        # Check if use double type MoE shared_experts(MindSpeed & Megatron).
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size and (args.moe_shared_expert_intermediate_size != args.n_shared_experts * (
                args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)):
            raise AssertionError('`n_shared_experts` cannot be used with `moe_shared_expert_intermediate_size` together. Please use one of them.')
        # Convert args to megatron shared_experts for init.
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * (
                args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)
