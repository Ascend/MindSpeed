# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import os
from functools import wraps
import torch
from megatron.training import get_args


def start_grad_sync_wrapper(function):
    @wraps(function)
    def wrapper(self, *args, **kwargs):
        argument = get_args()
        if argument.enable_high_availability:
            self.use_distributed_optimizer = False
            function(self, *args, **kwargs)
            self.use_distributed_optimizer = True
        else:
            function(self, *args, **kwargs)     
    return wrapper  


def pipe_register_grad_ready(self, param: torch.nn.Parameter):
    assert (self.overlap_grad_reduce), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
    from mindspeed.moe.pipe_experts import FLAG_GRAD_REDUCE
    if self.is_last_microbatch and FLAG_GRAD_REDUCE:
        bucket = self.param_to_bucket[param]
        bucket.register_grad_ready(param)
