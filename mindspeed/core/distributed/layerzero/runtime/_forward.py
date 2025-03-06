import functools
from itertools import chain
from collections import deque
import logging
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributed.utils import (
    _cast_forward_inputs,
    _p_assert,
    _apply_to_tensors
)
from torch.utils._pytree import tree_flatten
from torch.autograd.graph import register_multi_grad_hook
from mindspeed.core.distributed.layerzero.zero3.api import BackwardReduceScatter
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _assert_in_training_states,
    _is_composable,
    _ZeRO3State,
    TrainingState,
)

from mindspeed.core.distributed.layerzero.zero3.flat_param import FlatParamHandle, HandleTrainingState
from mindspeed.core.distributed.layerzero import constants
from ._shard import _unshard, _reshard, _pre_forward_backward_unshard, _post_forward_reshard, _post_backward_reshard, _get_handle_to_post_backward
from ._grad import _reduce_grad, _accumulate_grad, _pre_bwd_reload_full_prec_grad
from ._utils import _reset_flat_param_grad_info_if_needed
from .hook import register_multi_post_grad_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
BACKWARD_POST_QUEUE = deque()


@no_type_check
def _register_pre_backward_hooks(
    state: _ZeRO3State,
    module: nn.Module,
    outputs: Any,
    handle: FlatParamHandle,
) -> None:
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).

    Returns:
        Forward pass outputs with pre-backward hooks registered to tensors that
        require gradients.
    """
    # If there is no gradient computation, then there is no need for
    # pre-backward logic
    if not torch.is_grad_enabled():
        return outputs
    if state._is_root:
        state._post_backward_callback_queued = False  # only defined on the root

    if handle:
        handle._needs_pre_backward_unshard = False
        handle._ran_pre_backward_hook = False
        # Since these handles' `FlatParameter`s participated in a forward, we
        # conservatively assume that they will be used in the backward

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.register_hook(
                functools.partial(_pre_backward_hook, state, module, handle)
            )
            if handle:
                handle._needs_pre_backward_unshard = True
        return t

    return _apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hook(
    state: _ZeRO3State,
    handle: Optional[FlatParamHandle],
) -> None:
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not handle:
        return
    flat_param = handle.flat_param
    inp_tensors = [p for p in flat_param._tensors if p.requires_grad]
    hook_handle = register_multi_post_grad_hook(
        inp_tensors, functools.partial(_post_backward_ready_hook, state, handle)
    )
    flat_param._post_backward_hook_state = (
        None, hook_handle)  # type: ignore[attr-defined]


def _register_post_backward_reshard_only_hook(
    state: _ZeRO3State,
    handle: Optional[FlatParamHandle],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """
    Registers post-backward hooks to reshard flat parameters that do not
    require gradient. We register these using multi-post-grad hooks on the
    input activations to ensure that all gradients that may depend on the
    parameters have been computed before resharding.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    # Construct `inp_tensors` lazily to avoid CPU overhead in typical case
    # where each flat parameter requires gradient
    inp_tensors: Optional[List[torch.Tensor]] = None
    if not handle:
        return
    if handle.flat_param.requires_grad:
        return
    if inp_tensors is None:
        args_list, _ = tree_flatten(args)
        kwargs_list, _ = tree_flatten(kwargs)
        inp_tensors = [
            obj
            for obj in chain(args_list, kwargs_list)
            if torch.is_tensor(obj) and obj.requires_grad
        ]
    _p_assert(inp_tensors is not None, "Got None inp_tensor")
    hook_handle = register_multi_grad_hook(
        inp_tensors, functools.partial(_post_backward_reshard, state, handle)
    )
    handle.flat_param._post_backward_hook_state = (
        hook_handle,)


@no_type_check
def _register_post_backward_final_callback(
    state: _ZeRO3State, module: nn.Module
) -> None:
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
    _p_assert(
        state._is_root,
        "Only the root ZeRo3 instance should register the post-backward callback",
    )
    if state._post_backward_callback_queued:
        return
    _assert_in_training_states(state, [TrainingState.IDLE])
    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
    )


@no_type_check
def _pre_forward(
    state: _ZeRO3State,
    handle: Optional[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function(f"LayerZeRO3._pre_forward"):
        # For `fully_shard` + `checkpoint`, skip pre-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            return args, kwargs
        state.training_state = TrainingState.FORWARD_BACKWARD
        state._exec_order_data.record_pre_forward(handle, module.training)
        if handle:
            handle._training_state = HandleTrainingState.FORWARD

        with torch.autograd.profiler.record_function("Unshard Function"):
            if unshard_fn is not None:
                unshard_fn(state, handle)
        if handle:
            handle._use_unsharded_views(as_params=False)
        if constants.AUTO_CAST_INPUT and state.mixed_precision:
            # Recursively convert args and kwargs to specified precision.
            input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        return args, kwargs


@no_type_check
def _post_forward(
    state: _ZeRO3State,
    handle: Optional[FlatParamHandle],
    reshard_fn: Callable,
    module: nn.Module,
    inputs: Any,
    output: Any,
) -> Any:
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
    with torch.profiler.record_function(f"LayerZeRO3._post_forward"):
        # For `fully_shard` + `checkpoint`, skip post-forward logic in the
        if handle and handle._training_state != HandleTrainingState.FORWARD:
            return output
        #! adapt megatron AC to avoid free after forward
        if handle and not handle.enter_backward:
            state._exec_order_data.record_post_forward(handle)
            with torch.autograd.profiler.record_function("Reshard Function"):
                if reshard_fn is not None:
                    reshard_fn(state, handle)
            # Register pre-backward hooks to unshard the flat parameters for the
            # gradient computation (if needed)
        output = _register_pre_backward_hooks(state, module, output, handle)
        state.training_state = TrainingState.IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE
        return output


@no_type_check
def _pre_backward_hook(
    state: _ZeRO3State,
    module: nn.Module,
    handle: FlatParamHandle,
    grad,
    *unused: Any,
) -> Any:
    """
    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    Post Condition:
        parameter in unshard and unpadded, used for grad compute 
        grad is unshard and unpadded. 
    """
    # Only run the pre-backward hook once per group of handles involved in the
    # same module forward computation
    if handle and getattr(handle, "_ran_pre_backward_hook", False):
        return grad
    if handle:
        handle.enter_backward = True
    with torch.profiler.record_function(f"LayerZeRO3._pre_backward_hook"):
        # Queue the post-backward callback once for the root FSDP instance to
        # attach it to the outermost backward graph task so that it is called
        # after all backward calls complete
        if state._is_root and not state._post_backward_callback_queued:
            _register_post_backward_final_callback(state, module)
            _reset_flat_param_grad_info_if_needed(state._all_handles)
        elif handle:
            allowed_states = [TrainingState.IDLE]
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)

        state.training_state = TrainingState.FORWARD_BACKWARD
        # Queueing the post-backward callback is the only logic that is not
        # per-handle in the pre-backward hook, so we can return early here if
        # there are no handles.
        if not handle:
            return grad
        #! ensure that last handle has finished accumulate grad (backward) on cpu
        if len(BACKWARD_POST_QUEUE) > 0:
            (_last_state, _last_handle) = BACKWARD_POST_QUEUE.popleft()
            _post_backward_hook(_last_state, _last_handle)
        handle._training_state = HandleTrainingState.BACKWARD_PRE
        _register_post_backward_hook(state, handle)
        _pre_forward_backward_unshard(state, handle)
        _pre_bwd_reload_full_prec_grad(state, handle)
        #! alloc memory on default stream if not allocated
        handle.prepare_gradient_for_backward()
        handle._ran_pre_backward_hook = True
        return grad


@no_type_check
@torch.no_grad()
def _post_backward_ready_hook(
    state: _ZeRO3State,
    handle: FlatParamHandle,
    *unused: Any,
):
    if not handle:
        return
    BACKWARD_POST_QUEUE.append((state, handle))
    

@no_type_check
@torch.no_grad()
def _post_backward_hook(
    state: _ZeRO3State,
    handle: FlatParamHandle,
    *unused: Any,
):
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If no sync, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad`` attribute is the reduced sharded
    gradient.
    """
    flat_param = handle.flat_param
    handle.enter_backward = False

    with torch.autograd.profiler.record_function(
        f"LayerZeRO3._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # For multiple applications of reentrant AC across submodules sharing
        # the same `FlatParameter`, the post-backward hook may run multiple
        # times in one backward, in which case we permit the state to already
        # be in `BACKWARD_POST`.
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST

        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError("ZeRO3 does not support gradients of gradients")
        
        _post_backward_reshard(state, handle)
        _accumulate_grad(state, handle)
        reduce_scatter_sync_gradients(state, handle)
        handle._ran_post_backward_hook = True


def reduce_scatter_sync_gradients(
        state: _ZeRO3State,
        handle: FlatParamHandle):
    '''
    Performs a sync in zero1 process group
    '''
    with torch.autograd.profiler.record_function(f"Reduce Scatter Gradients"):
        if not state._sync_gradients:
            return
        flat_param = handle.flat_param
        if flat_param is not None and flat_param._post_backward_called:
            return
        flat_param._post_backward_called = True
        if state.backward_reduce_scatter == BackwardReduceScatter.BACKWARD_PRE:
            state.wait_critical_path_events()
        _reduce_grad(state, handle)


@no_type_check
@torch.no_grad()
def _post_backward_final_callback_no_sync(
    state: _ZeRO3State,
    module: nn.Module,
):
    if not state._is_root or state._sync_gradients:
        raise RuntimeError("The post-backward no sync callback should only be called \
            on the root FSDP instance without sync gradients")

    while len(BACKWARD_POST_QUEUE) > 0:
        (_last_state, _last_handle) = BACKWARD_POST_QUEUE.popleft()
        _post_backward_hook(_last_state, _last_handle)
        
    root_state: _ZeRO3State = state
    root_state._exec_order_data.next_iter_during_accumulation()
    for zero3_state in state._all_zero3_states:
        zero3_state.training_state = TrainingState.IDLE
        handle: FlatParamHandle = zero3_state._handle
        if handle:
            handle._ran_pre_backward_hook = False
            handle._ran_post_backward_hook = False
            handle._training_state = HandleTrainingState.IDLE
            handle.prev_iter_synced = False
    if handle._offload_grads:
        while True:
            offload_event = root_state._offload_event_queue._dequeue()
            if offload_event:
                (event, last_handle) = offload_event
                event.wait()
                last_handle.free_full_prec_grad()
            else:
                break
    root_state._post_backward_callback_queued = False


@no_type_check
@torch.no_grad()
def _post_backward_final_callback_sync_gradients(
    state: _ZeRO3State,
    module: nn.Module
):
    if not (state._is_root and state._sync_gradients):
        raise RuntimeError("The post-backward sync callback should \
            only be called on the root FSDP instance with sync gradients")
        
    while len(BACKWARD_POST_QUEUE) > 0:
        (_last_state, _last_handle) = BACKWARD_POST_QUEUE.popleft()
        _post_backward_hook(_last_state, _last_handle)
        
    root_state: _ZeRO3State = state
    root_state._exec_order_data.next_iter()
    for zero3_state in state._all_zero3_states:
        _catch_all_reshard(zero3_state)
        zero3_state.training_state = TrainingState.IDLE
        handle: FlatParamHandle = zero3_state._handle
        #! if post_backward is done, but flat_param has not reduce scatter
        if state.backward_reduce_scatter == BackwardReduceScatter.BACKWARD_PRE:
            if handle and handle._ran_post_backward_hook and not handle.flat_param._post_backward_called:
                reduce_scatter_sync_gradients(zero3_state, handle)
        if handle:
            handle._ran_pre_backward_hook = False
            handle._ran_post_backward_hook = False
            handle._needs_pre_backward_unshard = False
            handle._post_forward_index = None
            handle._training_state = HandleTrainingState.IDLE
            handle._prefetched = False
            handle._needs_param_sync = root_state._sync_gradients
            handle._param_synced = False
            handle._grad_synced = False
            #! free handle zero3 shard if _sync_gradients in reshard after backward cause next run we use zero1 shard
            handle.flat_param._zero3_shard = None
            handle.prev_iter_synced = True

        _finalize_params(zero3_state)
    while True:
        rs_event = root_state._rs_event_queue._dequeue()
        if rs_event:
            (rs, last_handle) = rs_event
            rs.wait()
            last_handle.free_full_prec_grad()
        else:
            break

    compute_stream = state._default_stream
    compute_stream.wait_stream(root_state._post_backward_stream)
    for handle in state._all_handles:
        flat_param = handle.flat_param
        if flat_param.requires_grad:
            handle.prepare_gradient_for_zero1()
    root_state._post_backward_callback_queued = False


@no_type_check
@torch.no_grad()
def _post_backward_final_callback(
    state: _ZeRO3State,
    module: nn.Module
):
    """
    This waits for the post-backward to finish and performs some final cleanup.
    This runs at the end of the entire backward pass and should only be called
    on the root FSDP instance.
    """
    if dist.get_rank() == 0:
        logger.info(
            f"_post_backward_final_callback Being Called and reset states")
    if state._sync_gradients:
        _post_backward_final_callback_sync_gradients(state, module)
    else:
        _post_backward_final_callback_no_sync(state, module)


@no_type_check
def _catch_all_reshard(
    state: _ZeRO3State,
) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
    # Wrap with a try-except to provide a more informative traceback if an
    # error is raised
    try:
        if state._handle:
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._zero1_shard.data_ptr()
                # If FSDP skipped using sharded views, then the flat parameter
                # still points to the sharded data, so we need to reshard to
                # use sharded views
                and not state._handle._skipped_use_sharded_views
            )
            if already_resharded:
                return
            _reshard(state, state._handle, True)
    except Exception as e:
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        raise e


@no_type_check
def _finalize_params(
    state: _ZeRO3State,
) -> None:
    """Finalizes the parameters before the next iteration.
    """
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    if hasattr(flat_param, "_post_backward_hook_state"):
        post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
        expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
        _p_assert(
            post_backward_hook_state_len == expected_post_backward_hook_state_len,
            f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
        )
        flat_param._post_backward_hook_state[-1].remove()
        delattr(flat_param, "_post_backward_hook_state")
    if flat_param.requires_grad:
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        flat_param._post_backward_called = False
