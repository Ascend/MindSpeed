# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import types
from functools import wraps

import torch
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.optimizer.virtual_optimizer.virtual_adam import (
    virtual_optimizer_step_impl,
    virtual_optimizer_replace,
    VirtualAllocator,
)


def virtual_optimizer_step(self, closure=None):
    if not hasattr(self, "virtual_allocator"):
        self.virtual_allocator = get_global_virtual_allocator()
    self.print_swap_flag = not hasattr(self, "print_swap_flag")
    with torch.no_grad():
        loss = virtual_optimizer_step_impl(self, closure)
    return loss


def _bind_raw_optimizer(optimizer):
    """Bind the virtual step to the optimizer instance created by MCore 0.18."""
    if optimizer is None:
        return

    if getattr(optimizer, "_mindspeed_virtual_optimizer_bound", False):
        if getattr(optimizer, "state", None):
            virtual_optimizer_replace(optimizer, get_global_virtual_allocator())
        return

    for group in getattr(optimizer, "param_groups", []):
        group.setdefault("amsgrad", False)
        group.setdefault("maximize", False)

    optimizer.step = types.MethodType(virtual_optimizer_step, optimizer)
    optimizer._mindspeed_virtual_optimizer_bound = True

    # Float16Optimizer and DistributedOptimizer initialize Adam states while
    # they are being constructed. Move those existing states immediately;
    # empty states will be allocated from swap memory by the first step.
    if getattr(optimizer, "state", None):
        virtual_optimizer_replace(optimizer, get_global_virtual_allocator())


def bind_virtual_optimizer(optimizer):
    """Find and bind all raw optimizers contained in an MCore optimizer result."""
    if optimizer is None:
        return optimizer
    if isinstance(optimizer, tuple):
        for item in optimizer:
            bind_virtual_optimizer(item)
        return optimizer

    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for item in chained:
            bind_virtual_optimizer(item)
        return optimizer

    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        bind_virtual_optimizer(inner)
    elif hasattr(optimizer, "param_groups") and hasattr(optimizer, "state"):
        _bind_raw_optimizer(optimizer)
    return optimizer


def get_optimizer_builder_wrapper(fn):
    """Adapt the MCore 0.18 optimizer factory without assuming an Adam class."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        optimizer = fn(*args, **kwargs)
        return bind_virtual_optimizer(optimizer)

    return wrapper


def get_global_virtual_allocator():
    """
    Get global virtual allocator.
    """
    args = get_args()
    if not hasattr(args, "virtual_allocator"):
        args.virtual_allocator = VirtualAllocator(
            get_pipeline_model_parallel_rank(),
            get_pipeline_model_parallel_world_size(),
            get_args().virtual_optimizer,
        )
    return args.virtual_allocator


def replace_swap_tensor_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        bind_virtual_optimizer(self)
        return res

    return wrapper
