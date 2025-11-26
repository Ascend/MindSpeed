# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu
from mindspeed.op_builder import MatmulAddOpBuilder

__all__ = ["npu_matmul_add_fp32"]

IS_A5 = None

matmul_add_op_builder = MatmulAddOpBuilder()


def npu_matmul_add_fp32(total_input, grad_output, grad):
    # 检查total_input的shape是否有维度为0
    for dim in total_input.shape:
        if dim == 0:
            return

    # 检查grad_output的shape是否有维度为0
    for dim in grad_output.shape:
        if dim == 0:
            return
    global IS_A5
    if IS_A5 is None:
        if "Ascend910_95" in torch_npu.npu.get_device_name():
            IS_A5 = True
        else:
            IS_A5 = False
    if IS_A5:
        grad.addmm_(grad_output.t(), total_input)
    else:
        matmul_add_ops = matmul_add_op_builder.load()
        matmul_add_ops.npu_matmul_add_fp32(grad_output, total_input, grad)


def npu_matmul_add_fp16(total_input, grad_output, grad):
    # 检查total_input的shape是否有维度为0
    for dim in total_input.shape:
        if dim == 0:
            return

    # 检查grad_output的shape是否有维度为0
    for dim in grad_output.shape:
        if dim == 0:
            return

    grad_weight = grad_output.t().matmul(total_input)
    grad.add_(grad_weight)
