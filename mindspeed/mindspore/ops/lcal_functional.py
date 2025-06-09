# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch

from megatron.core import mpu

import torch_npu


def matmul_all_reduce(self, input1, input2, output, bias=None):
    device = torch.distributed.get_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    comm_domain = str(device // tp_size)
    rank = device % tp_size
    torch_npu.matmul_all_reduce(input1, input2, bias, output, rank, tp_size, comm_domain)
    return output


def all_gather_matmul(self, input1, input2, output, bias=None):
    device = torch.distributed.get_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    comm_domain = str(device // tp_size)
    rank = device % tp_size
    torch_npu.all_gather_matmul(input1, input2, bias, output, rank, tp_size, comm_domain)
    return output


def all_gather_matmul_v2(self, input1, input2, output, comm_output, bias=None):
    device = torch.distributed.get_rank()
    if self.comm_config is None:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        comm_domain = str(device // tp_size)
        rank = device % tp_size
    else:
        tp_size = self.comm_config.rank_size
        comm_domain = self.comm_config.comm_domain
        rank = self.comm_config.rank_id
    torch_npu.all_gather_matmul_v2(input1, input2, bias, output, comm_output, rank, tp_size, comm_domain)
    return output, comm_output


def matmul_reduce_scatter(self, input1, input2, output, bias=None):
    device = torch.distributed.get_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    comm_domain = str(device // tp_size)
    rank = device % tp_size
    torch_npu.matmul_reduce_scatter(input1, input2, bias, output, rank, tp_size, comm_domain)
    return output


def pure_matmul(self, input1, input2, output, bias=None):
    device = torch.distributed.get_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    comm_domain = str(device // tp_size)
    rank = device % tp_size
    torch_npu.pure_matmul(input1, input2, bias, output, rank, tp_size, comm_domain)
    return output