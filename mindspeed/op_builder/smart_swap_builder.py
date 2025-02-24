# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
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
        args += ["-D_GLIBCXX_USE_CXX11_ABI=0", "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"]

        remove_arg = "-fvisibility=hidden"
        while remove_arg in args:
            args.remove(remove_arg)

        return args

    def extra_ldflags(self):
        flags = ["-L" + os.path.join(self._torch_npu_path, "lib"), "-ltorch_npu"]
        return flags
