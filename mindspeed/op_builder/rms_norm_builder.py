from mindspeed.op_builder.atb_builder import AtbOpBuilder


class RmsNormOpBuilder(AtbOpBuilder):
    OP_NAME = "rms_norm"

    def __init__(self):
        super(RmsNormOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/rms_norm.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp']
    