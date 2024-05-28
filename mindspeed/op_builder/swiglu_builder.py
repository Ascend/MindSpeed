from mindspeed.op_builder.atb_builder import AtbOpBuilder


class SwigluOpBuilder(AtbOpBuilder):
    OP_NAME = "swiglu"

    def __init__(self):
        super(SwigluOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/swiglu.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp']