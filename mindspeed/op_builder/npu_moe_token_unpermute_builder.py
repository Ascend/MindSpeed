# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.op_builder.builder import MindSpeedOpBuilder


class MoeTokenUnpermuteOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_moe_token_unpermute"

    def __init__(self):
        super(MoeTokenUnpermuteOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/npu_moe_token_unpermute.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        return args
