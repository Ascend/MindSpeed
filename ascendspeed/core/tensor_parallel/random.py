from functools import wraps
import torch
from torch import _C
from torch_npu.npu import _lazy_call, device as device_ctx_manager
from megatron import get_args
from megatron.core.tensor_parallel.utils import gather_split_1d_tensor
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.utils import safely_set_viewless_tensor_data
from torch.utils.checkpoint import detach_variable
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)


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


def backward(ctx, *args):
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
    detached_inputs = detach_variable(inputs)
    if global_args.optimize_recomp_communication_level > 0:
        if global_args.sequence_parallel:
            dim_size = list(args[0].size())
            dim_size[0] = dim_size[0] * get_tensor_model_parallel_world_size()
            allgather_grad = torch.empty(dim_size, dtype=args[0].dtype, device=torch.cuda.current_device())
            handle = torch.distributed._all_gather_base(allgather_grad, args[0].contiguous(),
                                                        group=get_tensor_model_parallel_group(),
                                                        async_op=True)
        with torch.enable_grad():
            global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_level
            outputs = ctx.run_function(*detached_inputs)
            global_args.optimize_recomp_communication_status = 0
        if global_args.sequence_parallel:
            handle.wait()
            args = tuple([allgather_grad])
    else:
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)

    # filter out non tensor outputs for backward pass
    outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
    torch.autograd.backward(outputs, args)
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return (None, None) + grads
