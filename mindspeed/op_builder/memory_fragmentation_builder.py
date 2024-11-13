import torch
import os

from mindspeed.op_builder.builder import MindSpeedOpBuilder


class MemoryFragmentationBuilder(MindSpeedOpBuilder):
    NAME = "MemoryFragmentation"
    _torch_path = None
    _python_path = None

    def __init__(self):
        from sysconfig import get_paths
        self._torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        self._python_path = os.path.dirname(os.path.abspath(get_paths().get('include')))
        super(MemoryFragmentationBuilder, self).__init__(self.NAME)

    def include_paths(self):
        paths = super().include_paths()
        paths += [
            os.path.join(self._torch_path, 'include'),
            os.path.join(self._torch_path, 'include/torch/csrc/api/include'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc/acl/'),
            os.path.join(self._python_path),
        ]
        return paths

    def sources(self):
        return ['ops/csrc/pluggable_allocator/memory_fragmentation/EventPool.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/CachingAllocatorConfig.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/DeviceCachingAllocator.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/PluggableAllocator.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/PluggableAllocatorFunctions.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/Decorator.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/Recorder.cpp',
                'ops/csrc/pluggable_allocator/memory_fragmentation/common.cpp']

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-s', '-D_FORTIFY_SOURCE=2', '-O2', "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"]
        return args

    def extra_ldflags(self):
        flags = [
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags