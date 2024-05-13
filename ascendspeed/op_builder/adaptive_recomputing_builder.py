import os

from ascendspeed.op_builder.builder import AscendSpeedOpBuilder


class AdaptiveRecomputingPluggableAllocatorBuilder(AscendSpeedOpBuilder):
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
        args = super().cxx_args()
        args += ['-D_GLIBCXX_USE_CXX11_ABI=0', "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"]
        return args

    def extra_ldflags(self):
        flags = [
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags
