# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Bytedance Inc. All rights reserved.
import logging
from functools import wraps
from collections import deque
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.parallel_state import (
    get_data_parallel_world_size,
    get_data_parallel_group,
    get_tensor_model_parallel_world_size,
    get_global_memory_buffer)
from megatron.legacy.model.transformer import FlashSelfAttention

from megatron.training import get_args
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel, logger
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
from megatron.core import parallel_state
import torch


@torch.no_grad()
def all_gather_param(param, wait_buffer):
    dp_size = get_data_parallel_world_size()
    group = get_data_parallel_group()
    dim_size = list(param.data.size())
    dim_size[0] = dim_size[0] * dp_size
    param.ds_tensor = param.data
    param.data = torch.empty(dim_size, dtype=param.data.dtype, device=torch.cuda.current_device())
    wait_buffer.append(torch.distributed._all_gather_base(param.data, param.ds_tensor.contiguous(), async_op=True, group=group))


@torch.no_grad()
def reduce_scatter_grad(param, wait_grad_buffer):
    dp_size = get_data_parallel_world_size()
    scale = 1.0
    if dp_size > 0 :
        scale = scale / dp_size
    param.full_grad.data *= scale
    group = get_data_parallel_group()
    param.grad_data_buffer = torch.empty(param.ds_tensor.shape, dtype=param.full_grad.dtype, device=torch.cuda.current_device())
    wait_grad_buffer.append(torch.distributed._reduce_scatter_base(param.grad_data_buffer, param.full_grad.data.contiguous(), async_op=True, group=group))


@torch.no_grad()
def release_param_data(param):
    param.data = param.ds_tensor


def wait_grad(param, wait_grad_buffer):
    wait_grad_buffer.popleft().wait()
    param.main_grad.add_(param.grad_data_buffer)
    param.grad_data_buffer = None
    param.full_grad = None
    param.grad = None


def set_model_fw_bw_hook(modules):
    wait_buffer = deque()
    wait_grad_buffer = deque()
    dp_size = get_data_parallel_world_size()
    if dp_size == 1:
        return 
    module_list = []
    fa_module = False
    for module in modules:
        fa_module |= isinstance(module, FlashSelfAttention)
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            module.pre_module_id = module.next_module_id = None
            module_list.append(module)
            if fa_module:
                # Send h_to_4h information in advance for communication masking.
                module.light_weight = True
                fa_module = False
    if len(module_list) > 0:
        module_list[0].zero_start = True
        module_list[-1].zero_end = True
    for i in range(len(module_list) - 1):
        module_list[i].next_module_id = i + 1
        module_list[i + 1].pre_module_id = i
    

    def forward_pre_hook(module, *arg):
        if hasattr(module, 'zero_start'):
            all_gather_param(module.weight, wait_buffer)
        wait_buffer.popleft().wait()
        if hasattr(module, 'light_weight'):
            return
        next_module_id = module.next_module_id
        if next_module_id is not None:
            next_module = module_list[next_module_id]
            all_gather_param(next_module.weight, wait_buffer)
            if hasattr(next_module, 'light_weight') and next_module.next_module_id is not None:
                all_gather_param(module_list[next_module.next_module_id].weight, wait_buffer)
        

    def forward_hook(module, *args):
        release_param_data(module.weight)


    def backward_pre_hook(module, *args):
        if hasattr(module, 'zero_end'):
            all_gather_param(module.weight, wait_buffer)
        wait_buffer.popleft().wait()
        if hasattr(module, 'light_weight'):
            return
        pre_module_id = module.pre_module_id
        if pre_module_id is not None:
            pre_module = module_list[pre_module_id]
            all_gather_param(pre_module.weight, wait_buffer)
            if hasattr(pre_module, 'light_weight') and pre_module.pre_module_id is not None:
                all_gather_param(module_list[pre_module.pre_module_id].weight, wait_buffer)


    def backward_hook(module, *arg):
        release_param_data(module.weight)
        reduce_scatter_grad(module.weight, wait_grad_buffer)
        if hasattr(module, 'light_weight'):
            return
        next_module_id = module.next_module_id
        if next_module_id is not None:
            next_module = module_list[next_module_id]
            if hasattr(next_module, 'light_weight') and next_module.next_module_id is not None:
                wait_grad(module_list[next_module.next_module_id].weight, wait_grad_buffer)
            wait_grad(next_module.weight, wait_grad_buffer)
        if hasattr(module, 'zero_start'):
            wait_grad(module.weight, wait_grad_buffer)

    for module in module_list:
        module.register_forward_pre_hook(hook=forward_pre_hook)
        module.register_forward_hook(hook=forward_hook)
        module.register_full_backward_pre_hook(hook=backward_pre_hook)
        module.register_full_backward_hook(hook=backward_hook)


def distributed_data_parallel_zero_grad_wrapper(function):
    @wraps(function)
    def distributed_data_parallel_zero_grad(self, *args, **kwargs):
        function(self, *args, **kwargs)
        for p in self.zero3_param:
            p.main_grad.data.zero_()
    return distributed_data_parallel_zero_grad
