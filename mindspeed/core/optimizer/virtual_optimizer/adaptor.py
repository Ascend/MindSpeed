# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from megatron.core.parallel_state import get_pipeline_model_parallel_rank, get_pipeline_model_parallel_world_size
from megatron.training import get_args
from mindspeed.core.optimizer.virtual_optimizer.virtual_adam import virtual_optimizer_step_impl, VirtualAllocator


def virtual_optimizer_step(self, closure=None):
    if not hasattr(self, "virtual_allocator"):
        self.virtual_allocator = VirtualAllocator(
            get_pipeline_model_parallel_rank(),
            get_pipeline_model_parallel_world_size(),
            get_args().virtual_optimizer)
    with torch.no_grad():
        loss = virtual_optimizer_step_impl(self, closure)
    return loss
