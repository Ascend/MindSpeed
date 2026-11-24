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

import os

from mindspeed.op_builder.builder import MindSpeedOpBuilder


class AdaptiveRecomputingPluggableAllocatorBuilder(MindSpeedOpBuilder):
    NAME = "AdaptiveRecomputing"
    _torch_path = None

    def __init__(self):
        import torch
        self._torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        super(AdaptiveRecomputingPluggableAllocatorBuilder, self).__init__(self.NAME)

    def include_paths(self):
        paths = super().include_paths()
        paths += [
            os.path.join(self._torch_path, 'include'),
            os.path.join(self._torch_path, 'include/torch/csrc/api/include'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc/acl/'),
        ]
        return paths

    def sources(self):
        return ['ops/csrc/pluggable_allocator/adaptive_recomputing/NpuCachingCustomAllocator.cpp']

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-s', '-D_FORTIFY_SOURCE=2', '-O2',
                "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"]
        return args

    def extra_ldflags(self):
        flags = [
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags
