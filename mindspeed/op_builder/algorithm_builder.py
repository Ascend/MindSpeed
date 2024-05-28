from mindspeed.op_builder.builder import MindSpeedOpBuilder


class AlgorithmOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "algorithm"

    def __init__(self):
        super(AlgorithmOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/algorithm/algorithm.cpp']
