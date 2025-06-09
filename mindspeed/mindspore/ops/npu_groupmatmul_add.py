# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
__all__ = ["npu_groupmatmul_add_fp32"]

import torch_npu


def npu_groupmatmul_add_fp32(x, dy, grouplist, grad):
    torch_npu.npu_groupmatmul_add_fp32(x, dy, grouplist, grad)
