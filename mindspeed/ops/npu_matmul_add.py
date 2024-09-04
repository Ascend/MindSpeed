# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.op_builder import MatmulAddOpBuilder
matmul_add_op_builder = MatmulAddOpBuilder()


def npu_matmul_add(total_input, grad_output, grad):
    matmul_add_ops = matmul_add_op_builder.load()
    matmul_add_ops.npu_matmul_add(grad_output, total_input, grad)
