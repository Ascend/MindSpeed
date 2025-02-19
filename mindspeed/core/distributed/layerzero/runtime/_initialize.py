from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple
import logging
import torch.nn as nn
from torch.distributed.utils import _p_assert
import torch.distributed as dist

import mindspeed.core.distributed.layerzero.zero3._traversal_utils as traversal_utils
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _assert_in_training_states,
    _ZeRO3State,
    TrainingState,
)
from ._utils import (
    _get_buffers_and_dtypes_for_computation,
    _cast_buffers_to_dtype_and_device,
)


@no_type_check
def _lazy_init(
    state: _ZeRO3State,
    root_module: nn.Module,
) -> _ZeRO3State:
    """
    Performs initialization lazily, typically right before the first forward
    pass. The laziness is needed to ensure that the parameter device/dtype and
    the FSDP hierarchy have finalized. This method's actual logic only runs on
    the root FSDP instance, which performs initialization for all non-root FSDP
    instances to avoid partial initialization.

    For the non-composable code path, ``state`` and ``root_module`` should be
    the same, namely the zero3 instance itself.
    """
    if state._is_root is not None:
        return None
    if not state._device_handle.is_available():
        # Allow the FSDP constructor to run even without CUDA but check this
        # once we start real execution
        raise RuntimeError("ZeRO3 does not support CPU only execution")
    # The following logic is only run on the root FSDP instance since it will
    # set `_is_root=False` for the non-root instances
    state._is_root = True
    _assert_in_training_states(state, [TrainingState.IDLE])
    _check_flat_params_on_expected_device(state, root_module)
    state._all_zero3_states = traversal_utils._get_zero3_states(root_module)
    _init_streams(state)
    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_computation(state, root_module)
    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, state.compute_device)
    state._exec_order_data.init(state, root_module, state.zero1_process_group)
    _share_state_and_init_handle_attrs(state, root_module)
    if dist.get_rank() == 0:
        logging.info(f"Root Layezero Contains {len(state._all_handles)} non-None handles")
    return state


def _check_flat_params_on_expected_device(state: _ZeRO3State, module: nn.Module):
    """
    Checks that all ``FlatParameter``s in ``module`` 's tree managed by
    ``state`` are on the expected device for *lazy initialization*.
    """
    for handle in traversal_utils._get_zero3_handles(module):
        if handle.flat_param.device != state.compute_device:
            raise RuntimeError(
                "An ZeRO3-managed module unexpectedly has parameters on "
                f"{handle.flat_param.device}. Make sure to move the module to "
                f"{state.compute_device} before training."
            )
            

@no_type_check
def _share_state_and_init_handle_attrs(
    root_state: _ZeRO3State,
    root_module: nn.Module,
) -> None:
    """
    Shares data structure state from the ``root_state`` to all zero3 states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    handle = root_state._handle
    if handle:
        handle.init_flat_param_attributes()
    root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
    for zero3_state in root_state._all_zero3_states:
        if zero3_state is root_state:
            continue
        _p_assert(
            zero3_state._is_root is None or not zero3_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been "
            "set yet or should have been set to `False`",
        )
        zero3_state._is_root = False
        zero3_state._unshard_stream = root_state._unshard_stream
        zero3_state._post_backward_stream = root_state._post_backward_stream
        zero3_state._pre_unshard_stream = root_state._pre_unshard_stream
        zero3_state._default_stream = root_state._default_stream
        zero3_state._offload_stream = root_state._offload_stream
        
        zero3_state._exec_order_data = root_state._exec_order_data
        zero3_state._free_event_queue = root_state._free_event_queue
        zero3_state._rs_event_queue = root_state._rs_event_queue
        zero3_state._offload_event_queue = root_state._offload_event_queue
        handle = zero3_state._handle
        if handle:
            handle.init_flat_param_attributes()


@no_type_check
def _init_streams(
    state: _ZeRO3State,
) -> None:
    """
    Initializes streams for overlapping communication, computation, and
    data transfers. The streams should be shared across zero3 instances.
    """
    if not (state._is_root and state._device_handle.is_available()):
        raise RuntimeError(f"state is not initialized or device not available")
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = 1
    mid_priority = 2
    low_priority = 3
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._device_handle.Stream(priority=mid_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._device_handle.Stream(priority=low_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._offload_stream = state._device_handle.Stream(priority=low_priority)
    state._pre_unshard_stream = state._device_handle.current_stream()