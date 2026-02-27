# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from functools import lru_cache
import torch
import torch_npu
from mindspeed.op_builder import MatmulAddOpBuilder

__all__ = ["npu_matmul_add_fp32"]

matmul_add_op_builder = MatmulAddOpBuilder()


@lru_cache
def is_a5():
    try:
        return "Ascend910_95" in torch_npu.npu.get_device_name() or "Ascend950" in torch_npu.npu.get_device_name()
    except Exception:
        return False


def npu_matmul_add_fp32(total_input, grad_output, grad):
    # 检查total_input的shape是否有维度为0
    for dim in total_input.shape:
        if dim == 0:
            return

    # 检查grad_output的shape是否有维度为0
    for dim in grad_output.shape:
        if dim == 0:
            return
    if is_a5():
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
