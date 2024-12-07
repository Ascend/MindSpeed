import os
import torch
from mindspeed.op_builder.builder import MindSpeedOpBuilder


class FusedEmaAdamWOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_apply_fused_ema_adamw"
    _torch_path = None

    def __init__(self):
        from sysconfig import get_paths
        self._torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        super(FusedEmaAdamWOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/npu_apply_fused_ema_adamw.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc',
                  os.path.join(self._torch_path, 'include'),
                  os.path.join(self._torch_path, 'include/torch/csrc/api/include'),
                  os.path.join(self._torch_npu_path, 'include/torch_npu/csrc/framework/utils'),
                  os.path.join(self._torch_npu_path, 'include/torch_npu/csrc/aten'),
                  ]
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += ['-Wno-narrowing']
        return args