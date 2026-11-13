# Copyright 2024 Huawei Technologies Co., Ltd.
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
import torch
from mindspeed.op_builder.builder import MindSpeedOpBuilder


class SmartSwapBuilder(MindSpeedOpBuilder):
    NAME = "smart_swap"
    _torch_path = None
    _python_path = None

    def __init__(self):
        from sysconfig import get_paths

        self._torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        self._python_path = os.path.dirname(os.path.abspath(get_paths().get("include")))
        super(SmartSwapBuilder, self).__init__(self.NAME)

    def sources(self):
        root_path = "ops/csrc/pluggable_allocator/smart_swap"
        files = [
            "./SwapBindings.cpp",
            "./NPUSwapManager.cpp",
            "./DeviceCachingAllocator.cpp",
            "./CachingAllocatorConfig.cpp",
            "./EventPool.cpp",
            "./PluggableAllocator.cpp",
            "./common.cpp",
        ]
        return [os.path.join(root_path, item) for item in files]

    def cxx_args(self):
        args = super().cxx_args()
        args += ["-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"]

        remove_arg = "-fvisibility=hidden"
        while remove_arg in args:
            args.remove(remove_arg)

        return args

    def extra_ldflags(self):
        flags = ["-L" + os.path.join(self._torch_npu_path, "lib"), "-ltorch_npu"]
        return flags
