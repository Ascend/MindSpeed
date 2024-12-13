# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from functools import wraps
import contextlib
from typing import Callable, Iterator, List, Optional, Union
import torch
from torch.autograd.variable import Variable
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.utils import get_model_config, get_model_type
from megatron.training import get_args
from megatron.core.pipeline_parallel.schedules import forward_step, backward_step, deallocate_output_tensor, check_first_val_step
from mindspeed.core.performance.auto_pipeline_perf.autopipeline_perf import profile_context
import mindspeed.core.training as training


def get_forward_backward_func_decorator(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        argument = get_args()
        pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        if pipeline_model_parallel_size > 1 and argument.automated_pipeline_perf and argument.optimized_mbs_list:
            forward_backward_func = optimized_forward_backward_pipelining
        else:
            forward_backward_func = get_forward_backward_func(*args, **kwargs)
        return forward_backward_func
    return wrapper


def forward_step_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        argument = get_args()
        if argument.automated_pipeline_perf and not (argument.optimized_mbs_list or argument.pp_schedule_list):
            torch.cuda.synchronize()
            start_time = time.time()
            output_tensor = fn(*args, **kwargs)
            torch.cuda.synchronize()
            profile_context["fwd_time"].append((time.time() - start_time) * 1000)
        else:
            output_tensor = fn(*args, **kwargs)
        return output_tensor

    return wrapper


def backward_step_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        argument = get_args()
        if argument.automated_pipeline_perf and not (argument.optimized_mbs_list or argument.pp_schedule_list):
            torch.cuda.synchronize()
            start_time = time.time()
            input_tensor_grad = fn(*args, **kwargs)
            torch.cuda.synchronize()
            profile_context["bwd_time"].append((time.time() - start_time) * 1000)
        else:
            input_tensor_grad = fn(*args, **kwargs)
        return input_tensor_grad
    return wrapper


def get_tensor_shapes():
    args = get_args()
    tensor_shapes = []
    mbs = args.optimized_mbs_list
    for m in mbs:
        tensor_shapes.append((args.seq_length // parallel_state.get_context_parallel_world_size() // parallel_state.get_tensor_model_parallel_world_size(), m, args.hidden_size))
    return tensor_shapes


def optimized_forward_backward_pipelining(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with reduced pipeline bubble.
    Returns dictionary with losses if the last stage, empty dict otherwise.
    """
    if isinstance(model, list):
        model = model[0]
    if isinstance(data_iterator, list):
        data_iterator = data_iterator[0]
    argument = get_args()
    config = get_model_config(model)
    model_type = get_model_type(model)
    tensor_shapes = get_tensor_shapes()
    cnt_fwd, cnt_bwd = 0, 0
    argument.mbs_idx = cnt_fwd
    argument.optimized_mbs_mode = True
    num_microbatches = len(argument.optimized_mbs_list)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Optimized pipeline parallelism does not support overlapping p2p communication"
        )

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensors = []
    forward_data_store = []
    rank = parallel_state.get_pipeline_model_parallel_rank()

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communication.recv_forward(config=config,
                                                      tensor_shape=tensor_shapes[cnt_fwd])
        argument.micro_batch_size = argument.optimized_mbs_list[cnt_fwd]
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(first_val_step, forward_only, i == 0),
        )
        p2p_communication.send_forward(output_tensor, config=config)
        cnt_fwd += 1
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(config=config,
                                                      tensor_shape=tensor_shapes[cnt_fwd])

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))
        argument.micro_batch_size = argument.optimized_mbs_list[cnt_fwd]
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            None,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
        )
        if forward_only:
            p2p_communication.send_forward(output_tensor, config=config)
            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(tensor_shapes=tensor_shapes[cnt_fwd], config=config)
        else:
            output_tensor_grad = \
                p2p_communication.send_forward_recv_backward(output_tensor,
                                                             tensor_shape=tensor_shapes[cnt_bwd], config=config)

        cnt_fwd += 1
        # Add input_tensor and output_tensor to end of list, then pop from the
        # start of the list for backward pass.
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        if forward_only:
            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(config=config,
                                                              tensor_shape=tensor_shapes[cnt_fwd])
        else:
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = \
                backward_step(input_tensor, output_tensor,
                              output_tensor_grad, model_type, config)

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, config=config)
            else:
                input_tensor = \
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad, tensor_shape=tensor_shapes[cnt_fwd], config=config)
        cnt_bwd += 1

    # Run cooldown backward passes.
    if not forward_only:
        for _ in range(num_warmup_microbatches):
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            output_tensor_grad = p2p_communication.recv_backward(
                tensor_shape=tensor_shapes[cnt_bwd], config=config)

            input_tensor_grad = \
                backward_step(input_tensor, output_tensor,
                              output_tensor_grad, model_type, config)
            p2p_communication.send_backward(input_tensor_grad, config)
            cnt_bwd += 1

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func([model])
    argument.optimized_mbs_mode = False
    argument.micro_batch_size = training.ORIGIN_MBS
    return forward_data_store
