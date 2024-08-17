# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
"""Pretrain utilities."""

import gc
from functools import wraps
import torch
import torch_npu
from megatron.training import get_args
from megatron.training import get_timers
from megatron.core import parallel_state
from megatron.training import print_rank_0
from megatron.training.arguments import parse_args
from mindspeed.core.memory.auto_pipeline.autopipeline import autopipeline_profiling
from mindspeed.core.memory.auto_pipeline.autopipeline_apply import apply_autopipeline
from mindspeed.core.memory.auto_pipeline.autopipeline_solver import solve_autopipeline, broadcast_policy_in_ranks, destroy_global_vars
from mindspeed.arguments import parse_args_wrapper


policy = None


def pretrain_decorator(pretrain):
    @wraps(pretrain)
    def wrapper(*args, **kwargs):
        new_parse_args = parse_args_wrapper(parse_args)
        argument = new_parse_args(kwargs.get('extra_args_provider'), False)
        if argument.automated_pipeline and not argument.num_layer_list:
            global policy
            context, policy = autopipeline_profiling(args[1], args[2], args[3],
                                             args[0], None, argument)
            if context:
                policy = solve_autopipeline(context)
                parallel_state.destroy_global_memory_buffer()
                parallel_state.destroy_model_parallel()
                destroy_global_vars()
                gc.collect()
                torch.cuda.empty_cache()
        pretrain(*args, **kwargs)
    return wrapper


def setup_model_and_optimizer_decorator(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(*args, **kwargs):
        global policy
        if policy:
            if torch.distributed.get_rank() == 0:
                broadcast_policy_in_ranks(0, policy)
            else:
                broadcast_policy_in_ranks(0)
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(*args, **kwargs)
        args = get_args()
        if args.recompute_module_list:
            apply_autopipeline(model)
        return model, optimizer, opt_param_scheduler
    return wrapper


def save_checkpoint_and_time_decorator(save_checkpoint_and_time):
    @wraps(save_checkpoint_and_time)
    def wrapper(*args, **kwargs):
        global_args = get_args()
        optimizer = args[2]
        if global_args.use_distributed_optimizer and global_args.overlap_param_gather:
            optimizer.disable_pre_hook()
        save_checkpoint_and_time(*args, **kwargs)
        if global_args.use_distributed_optimizer and global_args.overlap_param_gather:
            optimizer.enable_pre_hook()
    return wrapper
