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


class AnsCompressTensorFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("compress-dense", 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--compress-dense", type=str, default='disable',
                       choices=['disable', 'level0', 'level1'],
                       help='Compress activation in dense layer.')

    def validate_args(self, args):
        if args.compress_dense != "disable":
            import torch_npu
            if not hasattr(torch_npu, "npu_hans_encode") or not hasattr(torch_npu, "npu_hans_decode") \
                or not hasattr(torch_npu, "empty_with_swapped_memory"):  
                raise AssertionError("`--compress-dense` is invalid, please update the latest PTA version.")
            self.incompatible_check(args, "recompute_activation_function")
    
    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, "disable") != "disable":
            from mindspeed.core.memory.compress_dense.adaptor import mindspeed_compress_dense_forward
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward', mindspeed_compress_dense_forward)