# Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2024-2025 Advanced Micro Devices Inc.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""FP8 activation recompute checkpoint utilities.

Migrated from mindspeed/core/fp8/checkpoint.py.
Replaced mindspeed.core.fp8.* imports with TENPU direct calls.
"""

import warnings
from contextlib import AbstractContextManager, ContextDecorator
from typing import List, Callable, Tuple, Dict, Any, Union

import torch
from torch.utils.checkpoint import noop_context_fn, detach_variable

from megatron.core.tensor_parallel.random import _get_cuda_rng_state
from megatron.core.utils import safely_set_viewless_tensor_data
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.quantization.manager import (
    FP8GlobalStateManager as TENPU_FP8GlobalStateManager,
)

_FP8_ACTIVATION_RECOMPUTE_ENABLED = False
_FP8_ACTIVATION_RECOMPUTE_PHASE = False


# ---- Inlined from mindspeed/core/fp8/utils.py ----

def _get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
    """Return default TP group if tp_group is None."""
    if not torch.distributed.is_initialized():
        return None

    if tp_group is None:
        if is_expert:
            tp_group = get_tensor_model_parallel_group(check_initialized=check_initialized)
            # For expert TP, use get_expert_tensor_parallel_group if available
            try:
                from megatron.core.parallel_state import get_expert_tensor_parallel_group
                tp_group = get_expert_tensor_parallel_group(check_initialized=check_initialized)
            except ImportError:
                pass
        else:
            tp_group = get_tensor_model_parallel_group(check_initialized=check_initialized)
    return tp_group


def _split_tensor_into_1d_equal_chunks(tensor, new_buffer=False, tp_group=None):
    """Break a tensor into equal 1D chunks across tensor parallel ranks."""
    tp_group = _get_tensor_model_parallel_group_if_none(tp_group)
    partition_size = torch.numel(tensor) // tp_group.size()
    start_index = partition_size * tp_group.rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.npu.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def _gather_split_1d_tensor(tensor, tp_group=None):
    """Gather values from tensor model parallel ranks."""
    tp_group = _get_tensor_model_parallel_group_if_none(tp_group)
    numel_gathered = torch.numel(tensor) * tp_group.size()
    gathered = torch.empty(
        numel_gathered, dtype=tensor.dtype, device=torch.npu.current_device(), requires_grad=False
    )
    torch.distributed._all_gather_base(gathered, tensor, group=tp_group)
    return gathered


# ---- Context manager ----

class activation_recompute_forward(AbstractContextManager, ContextDecorator):
    """Context manager for FP8 activation recompute coordination.

    Sets global flags so FP8 modules know whether they are in the first
    forward pass or the recompute (backward) forward pass.
    """
    _is_first_fp8_module: List = []

    def __init__(self, activation_recompute: bool = False, recompute_phase: bool = False):
        super().__init__()
        self.activation_recompute = activation_recompute
        self.recompute_phase = recompute_phase

    def __enter__(self):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = self.activation_recompute
        _FP8_ACTIVATION_RECOMPUTE_PHASE = self.recompute_phase

        if self.activation_recompute and not self.recompute_phase:
            activation_recompute_forward._is_first_fp8_module.append(
                TENPU_FP8GlobalStateManager.quantization_state.is_first_fp8_module)
        if self.activation_recompute and self.recompute_phase:
            TENPU_FP8GlobalStateManager.quantization_state.is_first_fp8_module = \
                activation_recompute_forward._is_first_fp8_module.pop(0)

    def __exit__(self, *exc_details):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = False
        _FP8_ACTIVATION_RECOMPUTE_PHASE = False


def get_activation_recompute_contexts():
    """Returns context objects for the checkpointed forward pass and the forward recompute phase."""
    forward_ctx = activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=False,
    )
    recompute_ctx = activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=True,
    )
    return forward_ctx, recompute_ctx


def is_fp8_activation_recompute_enabled() -> bool:
    """Return global boolean for FP8 activation recompute enabled state."""
    return _FP8_ACTIVATION_RECOMPUTE_ENABLED


def in_fp8_activation_recompute_phase() -> bool:
    """Return global boolean for FP8 activation recompute phase."""
    return _FP8_ACTIVATION_RECOMPUTE_PHASE


# ---- Checkpoint function ----

def checkpoint(
    function: Callable,
    *args: Tuple[torch.Tensor, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """FP8-aware activation checkpoint.

    Adapted from TransformerEngine's checkpoint to support FP8
    activation recompute via activation_recompute_forward contexts.
    """
    global _USE_REENTRANT_ACTIVATION_RECOMPUTE
    _USE_REENTRANT_ACTIVATION_RECOMPUTE = kwargs.pop("use_reentrant", True)
    distribute_saved_activations = kwargs.pop("distribute_saved_activations", False)
    tp_group = kwargs.pop("tp_group", None)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Ensure backward compatibility.
    if (
        len(args) > 3
        and (isinstance(args[0], bool) or args[0] is None)
        and callable(args[1])
        and isinstance(args[2], None | torch.distributed.ProcessGroup)
    ):
        warnings.warn(
            "Passing non-tensor non-keyword arguments is deprecated and support will be removed in "
            "future releases of TransformerEngine. `distribute_saved_activations`, `tp_group`, and "
            "`get_rng_state_tracker` must be passed as keyword arguments to `checkpoint`.",
            DeprecationWarning,
            stacklevel=2,
        )
        distribute_saved_activations = args[0] if args[0] is not None else distribute_saved_activations
        get_rng_state_tracker = args[1]
        tp_group = args[2]
        args = args[3:]

    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)

    del determinism_check, debug
    if _USE_REENTRANT_ACTIVATION_RECOMPUTE:
        if distribute_saved_activations:
            assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
            tp_group = torch.distributed.GroupMember.WORLD if tp_group is None else tp_group

        return _CheckpointFunction.apply(
            function,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            context_fn,
            kwargs,
            *args,
        )

    if distribute_saved_activations:
        warnings.warn(
            "`distribute_saved_activations=True` has no effect when `use_reentrant=False`. "
            "The non-reentrant checkpoint implementation does not manually store forward "
            "inputs for the activation recompute in the backward pass, and instead leverages "
            "the autograd engine's pack/unpack hooks."
        )

    user_forward_ctx, user_recompute_ctx = context_fn()
    te_forward_ctx, te_recompute_ctx = get_activation_recompute_contexts()

    fp8 = TENPU_FP8GlobalStateManager.is_fp8_enabled()
    fp8_recipe = TENPU_FP8GlobalStateManager.get_fp8_recipe() if fp8 else None

    def recompute_fn(*args, **kwargs):
        with (
            torch.autograd.enable_grad(),
            te_recompute_ctx,
            user_recompute_ctx,
            fp8_autocast(enabled=fp8, fp8_recipe=fp8_recipe),
        ):
            function(*args, **kwargs)

    new_frame = _CheckpointFrame(
        recompute_fn,
        get_rng_state_tracker,
    )
    new_frame.cache_rng_states(forward=True)

    with _checkpoint_hook(new_frame, args, kwargs), te_forward_ctx, user_forward_ctx:
        out = function(*args, **kwargs)

    return out


if hasattr(torch, "_disable_dynamo"):
    checkpoint = torch._disable_dynamo(checkpoint)


class _CheckpointFunction(torch.autograd.Function):
    """Adapted from torch.utils.checkpoint with TE/MindSpeed FP8 support."""

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        distribute_saved_activations: bool,
        get_rng_state_tracker: Union[Callable, None],
        tp_group: Union[torch.distributed.ProcessGroup, None],
        context_fn: Union[Callable, None],
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()

        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
                outputs = run_function(*args, **kwargs)

        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                _split_tensor_into_1d_equal_chunks(args[0].data, tp_group=tp_group, new_buffer=True),
            )

        ctx.inputs = [arg if not torch.is_tensor(arg) else None for arg in args]
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs)

        fp8 = TENPU_FP8GlobalStateManager.is_fp8_enabled()
        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.tp_group = tp_group
        ctx.recompute_ctx = recompute_ctx
        ctx.fp8 = fp8
        ctx.fp8_recipe = TENPU_FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        ctx.kwargs = kwargs

        return outputs

    @staticmethod
    def backward(ctx, *args: Tuple[Union[torch.Tensor, None], ...]) -> Tuple[Union[torch.Tensor, None], ...]:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state

        inputs = tuple(t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.inputs))

        get_rng_state_tracker = ctx.get_rng_state_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                _gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
            )

        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        detached_inputs = detach_variable(inputs)
        with (
            torch.enable_grad(),
            ctx.recompute_ctx,
            activation_recompute_forward(activation_recompute=True, recompute_phase=True),
            fp8_autocast(enabled=ctx.fp8, fp8_recipe=ctx.fp8_recipe),
        ):
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True, this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
        return (None, None, None, None, None, None) + grads


class _CheckpointFrame:
    """Storage frame for forward RNG states and detached activations."""

    def __init__(self, recompute_fn: Callable, get_rng_state_tracker: Callable):
        self.recompute_fn = recompute_fn
        self.recomputed = []
        self.count = 0
        self.get_rng_state_tracker = get_rng_state_tracker
        self.fwd_rng_states = None
        self.bwd_rng_states = None

    def cache_rng_states(self, forward=True):
        rng_states = (
            torch.get_rng_state(),
            _get_cuda_rng_state(graph_safe=False),
        )
        if self.get_rng_state_tracker is not None:
            rng_states += (self.get_rng_state_tracker().get_states(),)

        if forward:
            self.fwd_rng_states = rng_states
        else:
            self.bwd_rng_states = rng_states

    def restore_rng_states(self, forward=True):
        from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state

        if forward:
            rng_states = self.fwd_rng_states
        else:
            rng_states = self.bwd_rng_states

        torch.set_rng_state(rng_states[0])
        _set_cuda_rng_state(rng_states[1], graph_safe=False)
        if self.get_rng_state_tracker is not None:
            self.get_rng_state_tracker().set_states(rng_states[2])


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):

    def __init__(self, frame):
        def pack_hook(x):
            frame.recomputed.append(x.detach())
            return x.detach()

        def unpack_hook(x):
            return x

        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):

    def __init__(self, frame, args, kwargs):
        def pack_hook(x):
            del x
            idx = frame.count
            frame.count += 1
            return idx

        def unpack_hook(idx):
            if not frame.recomputed:
                frame.cache_rng_states(forward=False)
                frame.restore_rng_states(forward=True)
                with _recomputation_hook(frame):
                    frame.recompute_fn(*args, **kwargs)
                frame.restore_rng_states(forward=False)

            activation = frame.recomputed[idx]
            frame.recomputed[idx] = None
            return activation

        super().__init__(pack_hook, unpack_hook)
