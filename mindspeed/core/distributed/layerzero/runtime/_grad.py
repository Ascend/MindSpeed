from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple

import torch.distributed as dist

from mindspeed.core.distributed.layerzero.zero3._common_utils import _ZeRO3State
from mindspeed.core.distributed.layerzero.zero3.flat_param import FlatParamHandle, HandleTrainingState
from ._utils import _div_if_needed
from ._shard import _is_last_order_forward


@no_type_check
def _reduce_grad(state: _ZeRO3State, handle: FlatParamHandle) -> None:
    """
    Runs gradient reduction, sharded gradient and the post-reduction callback.
    if accumulate grad, this func will not be called cause whole param unshard
    grad will be stored, rather than shard grad.
    """
    flat_param = handle.flat_param
    rs_event = state._rs_event_queue._dequeue()
    if rs_event:
        rs, last_hanlde = rs_event
        rs.wait()
        last_hanlde.free_full_prec_grad()
    padded_unsharded_grad, new_sharded_grad = handle._get_reduce_scatter_tensors()
    _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
    state._post_backward_stream.wait_stream(state._default_stream)
    with state._device_handle.stream(state._post_backward_stream):
        dist.reduce_scatter_tensor(
            new_sharded_grad,
            padded_unsharded_grad,
            group=handle._get_reduce_scatter_group(),
        )
        reduce_scatter_event = state._device_handle.Event()
        reduce_scatter_event.record()
        state._rs_event_queue.enqueue((reduce_scatter_event, handle))
    #! remove all-reduce logic and shard grad accumulation, and grad view logic
    handle.set_shard_grad(new_sharded_grad)


def offload_grad(
    state: _ZeRO3State, handle: FlatParamHandle
):
    if not handle:
        return
    # do not offload the last backward cause it is needed at first
    if _is_last_order_forward(state, handle):
        return
    off_event_handle = state._offload_event_queue._dequeue()
    if off_event_handle is not None:
        offload_event, last_handle = off_event_handle
        offload_event.wait()
        last_handle.free_full_prec_grad()
    state._offload_stream.wait_stream(state._default_stream)
    state._offload_stream.wait_stream(state._unshard_stream)
    with state._device_handle.stream(state._offload_stream):
        handle.offload_grad()
        event = state._device_handle.Event()
        event.record()
    state._offload_event_queue.enqueue((event, handle))


@no_type_check
def _pre_bwd_reload_full_prec_grad(
    state: "_ZeRO3State",
    handle: Optional["FlatParamHandle"],
) -> None:
    if not handle or handle._training_state != HandleTrainingState.BACKWARD_PRE:
        return

    if state._offload_grads:
        if not handle.already_load_full_prec_grad():
            handle.alloc_full_prec_grad()
        with state._device_handle.stream(state._offload_stream):
            handle.reload_full_prec_grad()
        handle._check_padded_unsharded(
            handle.flat_param._full_prec_grad_padded)


def _accumulate_grad(
    state: "_ZeRO3State",
    handle: Optional["FlatParamHandle"],
):
    if not handle or handle._training_state != HandleTrainingState.BACKWARD_POST:
        return
    if not handle.already_load_full_prec_grad():
        handle.alloc_full_prec_grad()
    if state._offload_grads:
        state._default_stream.wait_stream(state._offload_stream)
    #! accumulate grad on compute stream
    handle.accumulate_grad()
    handle.free_runtime_unshard_grad()

    if state._offload_grads and not state._sync_gradients:
        offload_grad(state, handle)
