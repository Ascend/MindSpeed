# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.op_builder.builder import MindSpeedOpBuilder


class RingAttentionUpdateOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_ring_attention_update"

    def __init__(self):
        super(RingAttentionUpdateOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/npu_ring_attention_update.cpp']

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
