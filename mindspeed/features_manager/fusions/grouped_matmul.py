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


class GroupedMatmulFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('grouped-matmul', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.fusions.grouped_matmul import Ops, grouped_gemm_is_available, \
            assert_grouped_gemm_is_available, get_device_capability
        patch_manager.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
        patch_manager.register_patch('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
                                     grouped_gemm_is_available)
        patch_manager.register_patch('megatron.core.transformer.moe.grouped_gemm_util.assert_grouped_gemm_is_available',
                                     assert_grouped_gemm_is_available)
        patch_manager.register_patch('torch.cuda.get_device_capability', get_device_capability)
