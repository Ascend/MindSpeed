# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from torch.autograd import recompute_instance

from megatron.core.tensor_parallel import get_cuda_rng_tracker

from megatron.core.utils import safely_set_viewless_tensor_data
from megatron.core.tensor_parallel import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks


def local_set_cuda_rng_state(new_state, device=-1): # -_set_cuda_rng_state , 有个问题，这个同时是helper function不知道会不会有问题
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    torch.cuda.set_rng_state(new_state)


def checkpoint_function_backward(ctx, *args):
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
    local_set_cuda_rng_state(ctx.fwd_cuda_rng_state) # 用这里的还是megatron ? 
    get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

    # Compute the forward pass.
    with torch.enable_grad():
        outputs, f_vjp = torch.autograd.vjp(ctx.run_function, *inputs)

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    local_set_cuda_rng_state(bwd_cuda_rng_state) # 用这里的还是megatron ?
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)

    # filter out non tensor outputs for backward pass
    grads = f_vjp(*args)
    return (None, None) + grads


def checkpoint_function_forward(ctx, run_function, distribute_saved_activations, *args):
    ctx.run_function = run_function
    ctx.distribute_saved_activations = distribute_saved_activations

    # Copy the rng states.
    ctx.fwd_cpu_rng_state = torch.get_rng_state()
    ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
    ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    recompute_instance.set_recompute(True)
    with torch.no_grad():
        outputs = run_function(*args)
    recompute_instance.set_recompute(False)

    # Divide hidden states across model parallel group and only keep
    # the chunk corresponding to the current rank.
    if distribute_saved_activations:
        ctx.input_0_shape = args[0].data.shape
        safely_set_viewless_tensor_data(
            args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
        )

    # Store everything.
    ctx.save_for_backward(*args)

    return outputs


