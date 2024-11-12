# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.op_builder import GroupMatmulAddOpBuilder

__all__ = ["npu_groupmatmul_add_fp32"]


groupmatmul_add_op_builder = GroupMatmulAddOpBuilder()


def npu_groupmatmul_add_fp32(x, dy, grouplist, grad):
    groupmatmul_add_ops = groupmatmul_add_op_builder.load()
    groupmatmul_add_ops.npu_groupmatmul_add_fp32(x, dy, grouplist, grad)
