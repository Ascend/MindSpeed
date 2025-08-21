# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from megatron.core import mpu
from torch.utils.checkpoint import detach_variable

from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state


class CheckpointFunctionWithoutOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, checkpoint, *args):
        with torch.no_grad():
            outputs = run_function(*args)

        # Store everything
        ctx.save_for_backward(*detach_variable(args))
        checkpoint.ctx = ctx

        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        outputs = ctx.outputs
        torch.autograd.backward(outputs, args)
        ctx.outputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput:
    def __init__(self, get_cuda_rng_tracker_func):
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None
        self.outputs = None
        self.get_cuda_rng_tracker = get_cuda_rng_tracker_func

    def checkpoint(self, run_function, distribute_saved_activations, *args):
        self.run_function = run_function

        if distribute_saved_activations:
            raise RuntimeError(
                "CheckpointFunctionWithoutOutput does not support "
                "distribute_saved_activations"
            )

        #Copy the rng states.
        self.fwd_cpu_rng_state = torch.get_rng_state()
        self.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        self.fwd_cuda_rng_state_tracker = self.get_cuda_rng_tracker().get_states()

        outputs = CheckpointFunctionWithoutOutput.apply(run_function, self, *args)
        self.outputs = outputs
        if isinstance(self.outputs, torch.Tensor):
            self.outputs = (self.outputs,)

        return outputs

    def discard_output(self):
        for output in self.outputs:
            output.untyped_storage().resize_(0)

    def recompute(self, _):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )

        # Store the current states.
        cur_cpu_rng_state = torch.get_rng_state()
        cur_cuda_rng_state = torch.cuda.get_rng_state()
        cur_cuda_rng_state_tracker = self.get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(self.fwd_cpu_rng_state)
        _set_cuda_rng_state(self.fwd_cuda_rng_state)
        self.get_cuda_rng_tracker().set_states(self.fwd_cuda_rng_state_tracker)

        with torch.enable_grad():
            outputs = self.run_function(*self.ctx.saved_tensors)
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(cur_cpu_rng_state)
        _set_cuda_rng_state(cur_cuda_rng_state)
        self.get_cuda_rng_tracker().set_states(cur_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for output, recomputation_output in zip(self.outputs, outputs):
            output_size = recomputation_output.untyped_storage().size()
            output.untyped_storage().resize_(output_size)
            with torch.no_grad():
                output.untyped_storage().copy_(recomputation_output.untyped_storage())

        self.ctx.outputs = outputs
        self.outputs = None
        self.ctx = None


def should_recompute(config, layer_number, num_recompute):
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = config.virtual_pipeline_model_parallel_size
    pp_size = config.pipeline_model_parallel_size

    if vpp_size is not None:
        layer_per_chunk = config.num_layers_per_virtual_pipeline_stage
    elif pp_size is not None:
        layer_per_chunk = config.num_layers // pp_size
    else:
        layer_per_chunk = config.num_layers

    if vpp_rank is None:
        vpp_rank = 0
    if vpp_size is None:
        vpp_size = 1
    recompute_priority = ((layer_number - 1) % layer_per_chunk) * vpp_size + vpp_rank
    full_recompute_layers = config.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True

        return False

    if num_recompute is None:
        return True

    return recompute_priority < num_recompute
