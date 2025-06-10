# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch_npu


def npu_matmul_add_fp32(total_input, grad_output, grad):
    # check if any dimension of total_input is 0
    for dim in total_input.shape:
        if dim == 0:
            return

    # check if any dimension of grad_output is 0
    for dim in grad_output.shape:
        if dim == 0:
            return

    torch_npu.npu_matmul_add_fp32(grad_output, total_input, grad)