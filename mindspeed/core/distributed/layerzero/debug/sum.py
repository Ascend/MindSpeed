# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.distributed as dist
from megatron.training.utils import print_rank_0


@torch.no_grad()
def print_total_grad_sum(params):
    for param in params:
        print_grad_sum(param)


@torch.no_grad()
def print_grad_sum(param):
    from megatron.core import mpu
    if getattr(param, "tensor_model_parallel", False):
        print_grad_sum_helper(param, mpu.get_data_parallel_group(with_context_parallel=True), "TP_shard")
    else:
        print_grad_sum_helper(param, dist.group.WORLD, "None TP")


@torch.no_grad()
def print_grad_sum_helper(param, group, msg):
    if param.grad is not None:
        g_sum = param.grad.contiguous().float().sum()
        p_sum = param.contiguous().float().sum()
    else:
        g_sum = torch.zeros([1]).float().to(param.device)
        p_sum = torch.zeros([1]).float().to(param.device)

    dist.all_reduce(g_sum, group=group)
    dist.all_reduce(p_sum, group=group)
    print_rank_0(f"{msg} Psum {p_sum.item()}, Gsum {g_sum.item()}")
