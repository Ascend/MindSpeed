import re
import os
from abc import ABC, abstractmethod
from torch.utils.cpp_extension import load
import torch_npu
import ascendspeed

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"


class AscendSpeedOpBuilder(ABC):
    _cann_path = None
    _torch_npu_path = None
    _cann_version = None
    _loaded_ops = {}

    def __init__(self, name):
        self.name = name
        self._cann_path = self.get_cann_path()
        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ or os.path.exists(os.environ[ASCEND_HOME_PATH]):
            return os.environ[ASCEND_HOME_PATH]
        return None

    def get_absolute_paths(self, paths):
        ascendspeed_path = os.path.abspath(os.path.dirname(ascendspeed.__file__))
        return [os.path.join(ascendspeed_path, path) for path in paths]

    @abstractmethod
    def sources(self):
        ...

    def include_paths(self):
        paths = [
            os.path.join(self._cann_path, 'include'),
            os.path.join(self._torch_npu_path, 'include'),
            os.path.join(self._torch_npu_path, 'third_party'),
            os.path.join(self._torch_npu_path, 'acl'),
            os.path.join(self._torch_npu_path, 'inc'),
        ]
        return paths

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-Wl,--disable-new-dtags,--rpath', '-s']
        return args

    def extra_ldflags(self):
        flags = [
            '-L' + os.path.join(self._cann_path, 'lib64'), '-lascendcl',
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags

    def load(self, verbose=True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        op_module = load(name=self.name,
                         sources=self.get_absolute_paths(self.sources()),
                         extra_include_paths=self.get_absolute_paths(self.include_paths()),
                         extra_cflags=self.cxx_args(),
                         extra_ldflags=self.extra_ldflags(),
                         verbose=verbose)
        __class__._loaded_ops[self.name] = op_module

        return op_module
