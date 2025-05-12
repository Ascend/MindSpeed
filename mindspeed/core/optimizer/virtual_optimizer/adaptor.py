# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps

import torch
from megatron.core.parallel_state import get_pipeline_model_parallel_rank, get_pipeline_model_parallel_world_size
from megatron.training import get_args
from mindspeed.core.optimizer.virtual_optimizer.virtual_adam import (
    virtual_optimizer_step_impl,
    virtual_optimizer_replace, 
    VirtualAllocator
)


def virtual_optimizer_step(self, closure=None):
    if not hasattr(self, "virtual_allocator"):
        self.virtual_allocator = get_global_virtual_allocator()
    self.print_swap_flag = not hasattr(self, "print_swap_flag")
    with torch.no_grad():
        loss = virtual_optimizer_step_impl(self, closure)
    return loss


def get_global_virtual_allocator():
    """
    Get global virtual allocator.
    """
    args = get_args()
    if not hasattr(args, "virtual_allocator"):
        args.virtual_allocator = VirtualAllocator(
            get_pipeline_model_parallel_rank(),
            get_pipeline_model_parallel_world_size(),
            get_args().virtual_optimizer)
    return args.virtual_allocator


def replace_swap_tensor_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        virtual_allocator = get_global_virtual_allocator()
        virtual_optimizer_replace(self.optimizer, virtual_allocator)
        return res
    return wrapper
