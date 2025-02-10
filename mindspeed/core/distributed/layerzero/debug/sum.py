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


def all_gather_into_flat_tensor(tensor: torch.Tensor, process_group):
    '''这个函数用于将不同rank上不同大小的tensor 聚合成一个大的flatTensor'''
    world_size = process_group.size()
    rank = dist.get_rank(process_group)
    
    # 如果tensor为None或没有元素，使用一个空 tensor
    if tensor is None or tensor.numel() == 0:
        local_tensor = torch.empty([0]).float().cuda()
    else:
        local_tensor = tensor.contiguous().flatten().float()
    
    # 获取所有进程中的 tensor 大小
    tensor_sizes = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
    if local_tensor.numel() > 0:
        tensor_sizes[rank] = torch.tensor([local_tensor.numel()], dtype=torch.int64).cuda()
    else:
        tensor_sizes[rank] = torch.tensor([0], dtype=torch.int64).cuda()
    dist.all_gather(tensor_sizes, tensor_sizes[rank], group=process_group)
    tensor_sizes = [int(size.item()) for size in tensor_sizes]
    
    # 找到最大 tensor 大小
    max_size = max(tensor_sizes)
    
    # 创建填充 tensor
    if max_size > 0:
        padding_tensor = torch.zeros(max_size, dtype=torch.float32, device=local_tensor.device).cuda()
    else:
        padding_tensor = torch.tensor([], dtype=torch.float32, device=local_tensor.device).cuda()
    
    # 将 local_tensor 填充到 padding_tensor
    if local_tensor.numel() > 0:
        padding_tensor[:local_tensor.numel()] = local_tensor
    
    # 创建列表来存储所有填充后的 tensor
    all_padding_tensors = [torch.zeros_like(padding_tensor).cuda() for _ in range(world_size)]
    
    # 收集所有填充后的 tensor
    dist.all_gather(all_padding_tensors, padding_tensor, group=process_group)
    
    # 拼接所有 tensor，去除填充部分
    flatten_tensor = torch.cat([t[:size] for t, size in zip(all_padding_tensors, tensor_sizes)], dim=0)
    return flatten_tensor