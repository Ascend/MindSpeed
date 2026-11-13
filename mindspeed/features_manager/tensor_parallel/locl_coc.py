# Copyright 2025 Huawei Technologies Co., Ltd.
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


class CoCFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('use-ascend-coc', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-ascend-coc", action='store_true', help="Use ascend coc")
        group.add_argument('--coc-mode', type=int, default=-1, help='coc-mode: 0=original, 1=rewrite, 2=coc default')
        group.add_argument('--coc-parallel-num', type=int, default=1, help='coc parallel num')
        group.add_argument('--coc-fused-kernel', action='store_true', help='use coc fused kernel')

    def validate_args(self, args):
        self.incompatible_check(args, 'unaligned_linear')
        if args.use_ascend_coc:
            if hasattr(args, 'use_ascend_mc2') and args.use_ascend_mc2:
                raise AssertionError('--mc2 and coc can not be used together')
            from mindspeed.ops.npu_matmul_add import check_npu_version, NPUVersion

            if check_npu_version(NPUVersion.A5) and args.coc_fused_kernel:
                raise AssertionError('coc fused kernel is not supported on Ascend 950.')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCColumnParallelLinear
        from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCRowParallelLinear

        patch_manager.register_patch(
            'megatron.core.tensor_parallel.layers.ColumnParallelLinear', MindSpeedCoCColumnParallelLinear
        )
        patch_manager.register_patch(
            'megatron.core.tensor_parallel.layers.RowParallelLinear', MindSpeedCoCRowParallelLinear
        )
