# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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
from functools import partial, wraps

import torch

from megatron.core import mpu
from megatron.training import get_args


def loss_func_for_async_allreduce(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function for async allreduce.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
        allreduce handle
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat(
        [torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)]
    )
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        if loss[0].isnan():
            return AssertionError(
                f"Rank {global_rank}: found NaN in local forward loss calculation. "
                f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
            )

    # Reduce loss for logging, which is different from megatron pretrain_gpt.py.
    reporting_loss = loss.clone().detach()
    allreduce_handle = torch.distributed.all_reduce(
        reporting_loss, group=mpu.get_data_parallel_group(), async_op=True
    )

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    ret = (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        ({"lm loss": (reporting_loss[0], reporting_loss[1])}, allreduce_handle),
    )
    return ret


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
        raise AssertionError(f"when using --async-log-allreduce , type of the first input must be {torch.distributed.Work}, but got {type(handle)}. Do you change your loss function? You can refer to {loss_func_for_async_allreduce.__module__}.{loss_func_for_async_allreduce.__qualname__}.")
    handle.wait()
    return val


def forward_step_wrapper(forward_step):
    def forward_step_func_wrapper(forward_step_func):
        @wraps(forward_step_func)
        def wrapper(*args, **kargs):
            forward_step_func_output: tuple[torch.Tensor, partial] = forward_step_func(*args, **kargs)
            if len(forward_step_func_output) < 2:
                raise AssertionError(f"When using --async-log-allreduce, make sure your '{forward_step_func.__module__+ '.' + forward_step_func.__qualname__}' function returns at least 2 outputs.")
            loss_func_with_mask: partial = forward_step_func_output[1]
            if not isinstance(loss_func_with_mask, partial):
                raise AssertionError(f"When using --async-log-allreduce make sure your '{forward_step_func.__module__+ '.' + forward_step_func.__qualname__}' function's 2nd return type is expected to be {partial}, but got type: {type(forward_step_func_output[1])}.")
            partial_args = loss_func_with_mask.args
            if len(partial_args) < 1:
                raise AssertionError(f"When using --async-log-allreduce make sure your '{forward_step_func.__module__+ '.' + forward_step_func.__qualname__}' function binds the loss function with at least one argument.")
            loss_mask = loss_func_with_mask.args[0]
            return forward_step_func_output[0], partial(loss_func_for_async_allreduce, loss_mask)
        return wrapper


    @wraps(forward_step)
    def wrapper(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=False,
        checkpoint_activations_microbatch=None,
        is_first_microbatch=False,
        current_microbatch=None,
    ):
        forward_step_func = forward_step_func_wrapper(forward_step_func)
        return forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            is_first_microbatch,
            current_microbatch,
        )

    return wrapper
