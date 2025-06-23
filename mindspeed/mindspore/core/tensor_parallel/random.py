# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
from functools import wraps
from typing import List, Union
import torch
from torch import _C
from torch_npu.npu import _lazy_call, device as device_ctx_manager
from torch.utils.checkpoint import _get_autocast_kwargs
from megatron.training import get_args
from megatron.core.tensor_parallel.utils import gather_split_1d_tensor
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.utils import safely_set_viewless_tensor_data
from torch.utils.checkpoint import detach_variable
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    is_pipeline_last_stage,
    get_virtual_pipeline_model_parallel_rank,
)
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager


def local_set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    torch.cuda.set_rng_state(new_state)


def checkpoint_function_backward(ctx, *args):
    global_args = get_args()
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError(
            "Checkpointing is not compatible with .grad(), "
            "please use .backward() if possible"
        )
    inputs = ctx.saved_tensors
    if ctx.distribute_saved_activations:
        safely_set_viewless_tensor_data(
            inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
        )

    # Store the current states.
    bwd_cpu_rng_state = torch.get_rng_state()
    bwd_cuda_rng_state = torch.cuda.get_rng_state()
    bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

    # Set the states to what it used to be before the forward pass.
    torch.set_rng_state(ctx.fwd_cpu_rng_state)
    local_set_cuda_rng_state(ctx.fwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

    # Compute the forward pass.
    with torch.enable_grad():
        outputs, f_vjp = torch.autograd.vjp(ctx.run_function, *inputs)

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    local_set_cuda_rng_state(bwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)

    grads = f_vjp(*args)

    return (None, None) + grads