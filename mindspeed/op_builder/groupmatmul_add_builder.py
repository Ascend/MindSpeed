# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from mindspeed.op_builder.atb_builder import AtbOpBuilder


class GroupMatmulAddOpBuilder(AtbOpBuilder):
    OP_NAME = "npu_groupmatmul_add_fp32"

    def __init__(self):
        super(GroupMatmulAddOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/groupmatmul_add.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/flop_counter/flop_counter.cpp']
    
