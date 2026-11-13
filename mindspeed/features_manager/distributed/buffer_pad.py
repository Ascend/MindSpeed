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


class BufferPadFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('param-and-grad-buffer-pad', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--param-and-grad-buffer-pad', type=int, default=None,
                           help='Use this argument to ensure that all buckets start at a memory address that is needed-byte. Set 512 for Ascend')

    def validate_args(self, args):
        if args.param_and_grad_buffer_pad and args.param_and_grad_buffer_pad <= 0:
            raise AssertionError('--param-and-grad-buffer-pad must be greater than 0')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.buffer_pad.adaptor import param_and_grad_buffer_init_pad
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
                                param_and_grad_buffer_init_pad)



