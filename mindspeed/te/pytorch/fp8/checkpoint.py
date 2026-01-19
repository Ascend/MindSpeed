import warnings
from contextlib import AbstractContextManager, ContextDecorator
from typing import List, Callable, Tuple, Dict, Any, Union

import torch
from torch.utils.checkpoint import noop_context_fn, detach_variable

from megatron.core.tensor_parallel.random import _get_cuda_rng_state
from megatron.core.utils import safely_set_viewless_tensor_data
from mindspeed.te.pytorch.fp8.fp8 import fp8_autocast
from mindspeed.te.pytorch.fp8.state_manager import FP8GlobalStateManager
from mindspeed.te.pytorch.utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

_FP8_ACTIVATION_RECOMPUTE_ENABLED = False
_FP8_ACTIVATION_RECOMPUTE_PHASE = False


class activation_recompute_forward(AbstractContextManager, ContextDecorator):
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
                FP8GlobalStateManager.IS_FIRST_FP8_MODULE
            )
        if self.activation_recompute and self.recompute_phase:
            FP8GlobalStateManager.IS_FIRST_FP8_MODULE = (
                activation_recompute_forward._is_first_fp8_module.pop(0)
            )

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
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_ENABLED


def in_fp8_activation_recompute_phase() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_PHASE


def checkpoint(
    function: Callable,
    *args: Tuple[torch.Tensor, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    .. warning::

        It is the user's responsibility to ensure identical behavior when calling
        :attr:`function` from the forward and backward pass. If different output is
        produced (e.g. due to global state), then the checkpointed version won't
        be numerically equivalent.

    .. warning::
        `use_reentrant=False` does not support early stopping, and will execute the entire forward
        pass for the checkpointed module when recomputing activations in the backward pass.

    Parameters
    ----------
    function : Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    distribute_saved_activations : bool, default = False
            if set to ``True`` and ``use_reentrant=True``, first tensor argument is distributed
            across the specified tensor parallel group (``tp_group``) before saving it for the
            backward pass. This has no effect when ``use_reentrant=False``.
    get_rng_state_tracker : Callable, default = None
            python callable which returns an instance of :class:`CudaRNGStatesTracker`.
    tp_group : ProcessGroup, default = None
            tensor parallel process group. Used only when ``distribute_saved_activations=True``
            and ``use_reentrant=True``. If ``None``, it falls back to the default group.
    use_reentrant : bool, default = True
            perform checkpointing in reentrant mode.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """
    # Pop out te.distributed.checkpoint() arguments
    global _USE_REENTRANT_ACTIVATION_RECOMPUTE
    _USE_REENTRANT_ACTIVATION_RECOMPUTE = kwargs.pop("use_reentrant", True)
    distribute_saved_activations = kwargs.pop("distribute_saved_activations", False)
    tp_group = kwargs.pop("tp_group", None)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Ensure backward compatibility.
    if (
        len(args) > 3
        and isinstance(args[0], bool)
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
        distribute_saved_activations = args[0]
        get_rng_state_tracker = args[1]
        tp_group = args[2]
        args = args[3:]

    # Trigger the native PyTorch checkpoint if the function is not or does not contain a
    # Transformer Engine module.
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)

    # Otherwise discard unused te.utils.checkpoint.checkpoint() arguments
    # and execute TE's own checkpointing
    # NOTE: This logic uses the TE checkpoint on all custom callable `function` handles because we
    #       cannot be sure there are no TE modules inside the function. It also means we might run
    #       the TE checkpoint for non-TE modules, so the TE checkpoint has to support a potential
    #       user context function.
    del determinism_check, debug
    if _USE_REENTRANT_ACTIVATION_RECOMPUTE:
        # If saved activations need to be distributed but there is no process group,
        # default to the world group.
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

    # Preserve the torch autocast contexts from the forward pass during recompute phase.

    fp8 = FP8GlobalStateManager.is_fp8_enabled()
    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None

    def recompute_fn(*args, **kwargs):
        with torch.autograd.enable_grad(), (
            te_recompute_ctx
        ), user_recompute_ctx, fp8_autocast(
            enabled=fp8, fp8_recipe=fp8_recipe
        ):
            function(*args, **kwargs)

    # Initialize a new checkpoint frame for each new forward pass.
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
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

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
        """Call forward function while saving state to be able to
        redo the computation later."""
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()

        # Preserve torch autocast context for the backward pass

        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
                outputs = run_function(*args, **kwargs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, tp_group=tp_group, new_buffer=True),
            )

        # Store everything.
        ctx.inputs = [arg if not torch.is_tensor(arg) else None for arg in args]
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs)

        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.tp_group = tp_group
        ctx.recompute_ctx = recompute_ctx
        ctx.fp8 = fp8
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        ctx.kwargs = kwargs

        return outputs

    @staticmethod
    def backward(
        ctx, *args: Tuple[Union[torch.Tensor, None], ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )
        from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
        inputs = tuple(
            t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.inputs)
        )

        get_rng_state_tracker = ctx.get_rng_state_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad(), ctx.recompute_ctx, activation_recompute_forward(
            activation_recompute=True, recompute_phase=True
        ), fp8_autocast(
            enabled=ctx.fp8, fp8_recipe=ctx.fp8_recipe
        ):
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

        # Set the states back to what it was at the start of this function.
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
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )

        # backward does not require entering autocast context because
        # backward implementations already retrieve fp8 recipe and
        # enablement from stored ctx.
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs
        )
        return (None, None, None, None, None, None) + grads


class _CheckpointFrame:
    """
    Storage frame for forward RNG states and detached activations from the forward recompute.
    """

    def __init__(self, recompute_fn: Callable, get_rng_state_tracker: Callable):
        self.recompute_fn = recompute_fn
        self.recomputed = []
        self.count = 0
        self.get_rng_state_tracker = get_rng_state_tracker
        self.fwd_rng_states = None
        self.bwd_rng_states = None

    def cache_rng_states(self, forward=True):
        """Cache fwd/bwd RNG states in the frame to restore later."""
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
        """Restore fwd/bwd RNG states that were previously cached into the frame."""
        from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
        if forward:
            rng_states = self.fwd_rng_states
        else:
            rng_states = self.bwd_rng_states

        torch.set_rng_state(rng_states[0])
        _set_cuda_rng_state(rng_states[1], graph_safe=False)
        if self.get_rng_state_tracker is not None:
            self.get_rng_state_tracker().set_states(rng_states[2])


class _recomputation_hook(
    torch.autograd.graph.saved_tensors_hooks
):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the activation recompute phase."""

    def __init__(self, frame):
        def pack_hook(x):
            """
            Packing hook for each recomputed activation passed into the `ctx.save_for_backward()`
            call in the forward recomputation.
            """
            frame.recomputed.append(x.detach())
            return x.detach()

        def unpack_hook(x):
            """
            No-op unpack hook that will never be called because the backward pass for the
            forward recomputation is never triggered.
            """
            return x

        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(
    torch.autograd.graph.saved_tensors_hooks
):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the checkpointed forward pass."""

    def __init__(self, frame, args, kwargs):
        def pack_hook(x):
            """
            Packing hook for each tensor passed into `ctx.save_for_backward()` call in the
            forward pass. Since this is the first forward pass, we discard the tensor and instead
            pack a placeholder tensor index into the autograd engine context.
            """
            del x
            idx = frame.count
            frame.count += 1
            return idx

        def unpack_hook(idx):
            """
            Unpacking hook for each tensor that comes out of the `ctx.saved_tensors` call in the
            backward pass. The first time this is called, the _recomputation_hook will save all the
            activation tensors from `ctx.save_for_backward()` in the forward recomputation into the
            _CheckpointFrame. Subsequent calls will simply return the already recomputed activation
            tensor at the given index of the _CheckpointFrame storage.
            """

            if not frame.recomputed:
                # Store current RNG states in the backward pass
                frame.cache_rng_states(forward=False)

                # Set RNG states to what we saved before the forward pass
                frame.restore_rng_states(forward=True)

                # Recompute the forward pass
                with _recomputation_hook(frame):
                    frame.recompute_fn(*args, **kwargs)

                # Restore RNG states back to the backward pass
                frame.restore_rng_states(forward=False)

            # Return the already recomputed activation tensor at the given index
            activation = frame.recomputed[idx]
            frame.recomputed[idx] = None
            return activation

        super().__init__(pack_hook, unpack_hook)
