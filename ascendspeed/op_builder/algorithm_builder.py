from ascendspeed.op_builder.builder import AscendSpeedOpBuilder


class AlgorithmOpBuilder(AscendSpeedOpBuilder):
    OP_NAME = "algorithm"

    def __init__(self):
        super(AlgorithmOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/algorithm/algorithm.cpp']
