# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from mindspeed.op_builder.atb_builder import AtbOpBuilder


class MatmulAddOpBuilder(AtbOpBuilder):
    OP_NAME = "npu_matmul_add_fp32"

    def __init__(self):
        super(MatmulAddOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/matmul_add.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp']
    
