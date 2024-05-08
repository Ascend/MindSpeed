from mindspeed.op_builder.builder import MindSpeedOpBuilder


class NpuDropoutAddLayerNormOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_dropout_add_layer_norm"

    def __init__(self):
        super(NpuDropoutAddLayerNormOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/npu_dropout_add_layer_norm.cpp']

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
