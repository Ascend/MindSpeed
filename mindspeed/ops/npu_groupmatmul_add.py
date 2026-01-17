# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu
from mindspeed.op_builder import GroupMatmulAddOpBuilder

__all__ = ["npu_groupmatmul_add_fp32"]


groupmatmul_add_op_builder = GroupMatmulAddOpBuilder()


def is_a5():
    try:
        return "Ascend910_95" in torch_npu.npu.get_device_name()
    except Exception:
        return false


def npu_groupmatmul_add_fp32(x, dy, grouplist, grad):
    if is_a5():
        torch_npu.npu_grouped_matmul_add_(grad, x, dy, grouplist)
    else:
        groupmatmul_add_ops = groupmatmul_add_op_builder.load()
        groupmatmul_add_ops.npu_groupmatmul_add_fp32(x, dy, grouplist.to('npu'), grad)
