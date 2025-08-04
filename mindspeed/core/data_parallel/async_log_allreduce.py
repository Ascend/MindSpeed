# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gc
from functools import partial, wraps

import torch

from megatron.core import mpu
from megatron.training import get_args
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_timers
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.utils import (
    unwrap_model,
    reduce_max_stat_across_model_parallel_group,
    logical_and_across_model_parallel_group)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training.training import cuda_graph_set_manual_hooks, cuda_graph_capture


def get_async_reduced_loss_value(x, key):
    """
    Retrieves the reduced loss value after waiting for the completion of the async all-reduce operation.

    Args:
    x (tuple): A tuple containing two elements:
               - A dictionary where the key corresponds to the loss value.
               - A `torch.distributed.Work` object used for waiting for the completion of the all-reduce operation.
    key (str): The key used to access the loss value from the dictionary.

    Returns:
    val: The loss value retrieved from the dictionary using the specified key. The type of `val` depends on the type stored in the dictionary for the given key.

    Raises:
    AssertionError: If the second element of `x` is not of type `torch.distributed.Work`, an assertion error is raised.
    """
    # Wait until the loss allreduce execution is complete.
    # In most cases, the loss allreduce has already completed when the program execution reaches this point.

    val = x[0][key]
    handle = x[1]
    if not isinstance(handle, torch.distributed.Work):
        raise AssertionError(
            f"when using --async-log-allreduce , type of the first input must be {torch.distributed.Work}, but got {type(handle)}.")
    handle.wait()

    return val


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # CUDA Graph capturing only executes once, when it's the first training iteration.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        cuda_graph_capture(model, config, args)

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Collect garbage and empty unused memory.
        gc.collect()
        torch.cuda.empty_cache()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
            args.micro_batch_size * \
            args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # Set the manual hooks when CUDA Graphs are enabled.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            cuda_graph_set_manual_hooks(model)

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        if args.async_log_allreduce:
            # when async_log_allreduce is on, loss_reduced is list[tuple(dict,torch.distributed.group)]
            losses_reduced_keys = losses_reduced[0][0].keys()
        else:
            losses_reduced_keys = losses_reduced[0].keys()

        for key in losses_reduced_keys:
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                if args.async_log_allreduce:
                    val = get_async_reduced_loss_value(x, key)
                else:
                    val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
