from ascendspeed.op_builder.atb_builder import AtbOpBuilder


class LcalOpBuilder(AtbOpBuilder):
    OP_NAME = "lcal"

    def __init__(self):
        super(LcalOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/lcal_coc.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp']
