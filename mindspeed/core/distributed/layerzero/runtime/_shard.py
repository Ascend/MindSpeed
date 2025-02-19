import logging

from enum import auto, Enum
from typing import Any, no_type_check, Optional, Set, Tuple, TYPE_CHECKING

import torch
from torch.distributed.utils import _p_assert
import torch.distributed as dist
from mindspeed.core.distributed.layerzero.zero3.api import BackwardPrefetch
from mindspeed.core.distributed.layerzero.zero3.flat_param import HandleTrainingState
if TYPE_CHECKING:
    from mindspeed.core.distributed.layerzero.zero3._common_utils import _ZeRO3State
    from mindspeed.core.distributed.layerzero.zero3.flat_param import FlatParamHandle

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()


@no_type_check
def _unshard(
    state: "_ZeRO3State",
    handle: "FlatParamHandle",
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handle or not handle.needs_unshard():
        return

    with state._device_handle.stream(pre_unshard_stream):
        handle.pre_unshard()

    unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function(
                "LayerZeRO3.rate_limiter"
            ):
                event.synchronize()
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()


@no_type_check
def _reshard(
    state: "_ZeRO3State",
    handle: "FlatParamHandle",
    free_unsharded_flat_param: bool,
):
    """
    Reshards the handle. ``free_unsharded_flat_param`` indicates whether to
    free the handle's padded unsharded flat parameter.
    """
    handle.reshard(free_unsharded_flat_param)
    if state.limit_all_gathers and free_unsharded_flat_param:
        free_event = state._device_handle.Event()
        free_event.record()
        state._free_event_queue.enqueue(free_event)
    # Since we prefetch entire handles keys at a time, conservatively mark
    # the entire key as no longer prefetched once we free at least one
    if free_unsharded_flat_param:
        handle._prefetched = False
    else:
        handle._prefetched = True


@no_type_check
def _pre_forward_backward_unshard(
    state: "_ZeRO3State",
    handle: Optional["FlatParamHandle"],
) -> None:
    """Unshards parameters in the pre-forward.
    1. check handle exists
    2. check zero1 synced params to zero3
    3. check zero3 prefetched
    4. prefetch next layer
        modified  _unshard func, which is called at each all-gather

    """
    if not handle:
        return
    # If the handles have been prefetched, then there is no need to call
    # `_unshard()` again
    if handle._training_state not in [HandleTrainingState.FORWARD, HandleTrainingState.BACKWARD_PRE]:
        return

    in_forward = handle._training_state == HandleTrainingState.FORWARD
    stage = "forward" if in_forward else "backward"
    guard_state = f"_needs_pre_{stage}_unshard"
    if in_forward or getattr(handle, guard_state):
        _unshard(
            state,
            handle,
            state._unshard_stream,
            state._pre_unshard_stream
        )
        setattr(handle, guard_state, False)
        state._default_stream.wait_stream(state._unshard_stream)
        handle._check_unsharded(handle.flat_param.data)

    _prefetch_mode = _PrefetchMode.FORWARD if handle._training_state == HandleTrainingState.FORWARD else _PrefetchMode.BACKWARD
    with torch.profiler.record_function(
        f"LayerZeRO3._pre_{stage}_prefetch"
    ):
        _prefetch_handle(state, handle, _prefetch_mode)


def _is_last_order_forward(
    state: "_ZeRO3State",
    handle: "FlatParamHandle"
) -> bool:
    return handle._post_forward_index == len(state._exec_order_data.all_handles) - 1


@no_type_check
def _post_forward_reshard(
    state: "_ZeRO3State",
    handle: "FlatParamHandle",
) -> None:
    """Reshards parameters in the post-forward.
    """
    if not handle:
        return
    free_unsharded_flat_param = not _is_last_order_forward(state, handle)
    with torch.profiler.record_function(
        "LayerZeRO3._post_forward_reshard"
    ):
        _reshard(state, handle, free_unsharded_flat_param)


def _post_backward_reshard(
    state: "_ZeRO3State",
    handle: "FlatParamHandle",
    *unused: Any,
) -> None:
    free_unsharded_flat_param = not (
        handle._pre_forward_order_index == 0 and not state._sync_gradients)
    with torch.profiler.record_function(
        "LayerZeRO3._post_backward_reshard"
    ):
        _reshard(state, handle, free_unsharded_flat_param)

    with torch.profiler.record_function(
        "LayerZeRO3._post_backward_prefetch"
    ):
        _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)


@no_type_check
def _prefetch_handle(
    state: "_ZeRO3State",
    current_handle: Optional["FlatParamHandle"],
    prefetch_mode: _PrefetchMode,
) -> None:
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
    if not current_handle:
        return
    handle = _get_handle_to_prefetch(state, current_handle)
    if not handle:
        return
    # Temporarily emulate the training state while calling `_unshard` to
    # ensure the correct `as_params` for `_use_unsharded_views()`
    prev_training_state = handle._training_state
    if prefetch_mode == _PrefetchMode.BACKWARD:
        handle._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode == _PrefetchMode.FORWARD:
        if handle.enter_backward:
            return
        handle._training_state = HandleTrainingState.FORWARD
    else:
        raise ValueError(f"Invalid prefetch mode on rank {state.zero3_rank}: {prefetch_mode}")
    # Prefetch the next set of handles without synchronizing to allow
    # the sync to happen as late as possible to maximize overlap
    _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._training_state = prev_training_state
    handle._prefetched = True


@no_type_check
def _get_handle_to_prefetch(
    state: "_ZeRO3State",
    current_handle: "FlatParamHandle",
) -> "FlatParamHandle":
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    training_state = _get_training_state(current_handle)
    valid_training_states = (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
        HandleTrainingState.FORWARD,
    )
    _p_assert(
        training_state in valid_training_states,
        f"Prefetching is only supported in {valid_training_states} but "
        f"currently in {training_state}",
    )
    eod = state._exec_order_data
    target_handle: Optional["FlatParamHandle"] = None
    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        target_handle_candidate = eod.get_handle_to_backward_prefetch(
            current_handle)
        if (
            target_handle_candidate
            # and target_handle_candidate._needs_pre_backward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handle_candidate = eod.get_handle_to_forward_prefetch(
            current_handle)
        if (
            target_handle_candidate
            # and target_handle_candidate._needs_pre_forward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None

    return target_handle


def _get_training_state(
    handle: "FlatParamHandle",
) -> HandleTrainingState:
    """Returns the training state of the handles in ``handle``."""
    _p_assert(handle, "Expects a non-empty handle")
    return handle._training_state


@no_type_check
def _get_handle_to_post_backward(
    state: "_ZeRO3State",
    current_handle: "FlatParamHandle",
) -> "FlatParamHandle":
    """
    Returns the last handle to do post_backward reduce scatter, where ``current_handle`` represents the current module.
    """
    eod = state._exec_order_data
    target_handle: Optional["FlatParamHandle"] = None
    target_handle = eod.get_handle_to_post_backward(current_handle)
    if target_handle:
        return [handle for handle in target_handle
                if (_get_training_state(handle) == HandleTrainingState.BACKWARD_POST) and not handle.flat_param._post_backward_called]
    else:
        return None
