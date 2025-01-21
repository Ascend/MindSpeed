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


def _set_cuda_rng_state(new_state, device=-1):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


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
    _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

    # Compute the forward pass.
    flops_counter = None
    if global_args.op_cal_tflops:
        from mindspeed.core.training import get_flops_counter
        flops_counter = get_flops_counter()
        flops_counter.pause()

    detached_inputs = detach_variable(inputs)
    from mindspeed.auto_tuning.module.parse.recompute_parser import get_recompute_parser, call_hook_func
    recompute_parser = get_recompute_parser()

    if (
            recompute_parser.skip_profiling_step <= recompute_parser.profiling_step <= recompute_parser.stop_profiling_step
            and os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE'):
        call_hook_func()
    with torch.enable_grad():
        outputs = ctx.run_function(*detached_inputs)
    # remove hook
    for hook_handle in recompute_parser.modules_hooks:
        hook_handle.remove()
    recompute_parser.modules_hooks.clear()

    if global_args.op_cal_tflops:
        flops_counter.resume()

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)

    # filter out non tensor outputs for backward pass
    outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]) and x[0].grad_fn is not None, zip(outputs, args)))
    torch.autograd.backward(outputs, args)
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return (None, None) + grads


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
    def __init__(self):
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None
        self.outputs = None

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
        self.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

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
        cur_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(self.fwd_cpu_rng_state)
        _set_cuda_rng_state(self.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(self.fwd_cuda_rng_state_tracker)

        with torch.enable_grad():
            outputs = self.run_function(*self.ctx.saved_tensors)
        self.run_function = None
        self.fwd_cpu_rng_state = None
        self.fwd_cuda_rng_state = None
        self.fwd_cuda_rng_state_tracker = None

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(cur_cpu_rng_state)
        _set_cuda_rng_state(cur_cuda_rng_state)
        get_cuda_rng_tracker().set_states(cur_cuda_rng_state_tracker)

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



class RngStateContext:
    def __init__(self, cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
        self.fwd_cpu_rng_state = cpu_rng_state
        self.fwd_cuda_rng_state = cuda_rng_state
        self.fwd_cuda_rng_state_tracker = cuda_rng_state_tracker


class CheckpointFunctionRipipe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        fwd_rng_state = RngStateContext(torch.get_rng_state(), torch.cuda.get_rng_state(), get_cuda_rng_tracker().get_states())
        with torch.no_grad():
            outputs = run_function(*args)

        # Store everything.
        ctx.detached_inputs = detach_variable(args)

        def recompute():
            # Store the current states.
            bwd_cpu_rng_state = torch.get_rng_state()
            bwd_cuda_rng_state = torch.cuda.get_rng_state()
            bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

            # Set the states to what it used to be before the forward pass.
            torch.set_rng_state(fwd_rng_state.fwd_cpu_rng_state)
            _set_cuda_rng_state(fwd_rng_state.fwd_cuda_rng_state)
            get_cuda_rng_tracker().set_states(fwd_rng_state.fwd_cuda_rng_state_tracker)

            with torch.enable_grad():
                outputs = run_function(*ctx.detached_inputs)
            ctx.outputs = outputs

            # Set the states back to what it was at the start of this function.
            torch.set_rng_state(bwd_cpu_rng_state)
            _set_cuda_rng_state(bwd_cuda_rng_state)
            get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)
        if get_pipeline_checkpoint_manager().do_pre_recompute:
            get_pipeline_checkpoint_manager().add_recompute(recompute)
        ctx.recompute_func = recompute

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        if not hasattr(ctx, 'outputs'):
            if get_pipeline_checkpoint_manager().do_pre_recompute:
                global_args = get_args()
                vpp_rank = get_virtual_pipeline_model_parallel_rank()
                # For last vpp chunk of last pp stage, we don't advance its recomputation.
                if global_args.recompute_in_advance and is_pipeline_last_stage():
                    get_pipeline_checkpoint_manager().recompute_next(vpp_rank)
                if not hasattr(ctx, 'outputs'):
                    raise RuntimeError(f"rank-{torch.distributed.get_rank()}: recompute is not done")
            else:
                ctx.recompute_func()

        outputs = ctx.outputs
        detached_inputs = ctx.detached_inputs
        ctx.outputs = None
        ctx.detached_inputs = None
        ctx.recompute_func = None

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads


def checkpoint_wrapper(checkpoint):
    @wraps(checkpoint)
    def wrapper(function, distribute_saved_activations, *args):
        if not get_pipeline_checkpoint_manager().open_ri_pipe:
            return checkpoint(function, distribute_saved_activations, *args)
        if not get_pipeline_checkpoint_manager().chunk_do_recompute:
            return function(*args)

        if distribute_saved_activations:
            raise RuntimeError("no distributed")

        return CheckpointFunctionRipipe.apply(function, distribute_saved_activations, *args)

    return wrapper