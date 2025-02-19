from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.utils import (
    _p_assert,
    _apply_to_tensors
)

import mindspeed.core.distributed.layerzero.zero3._traversal_utils as traversal_utils
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _assert_in_training_states,
    _get_module_zero3_state,
    _no_dispatch_record_stream,
    clean_tensor_name,
    _ZeRO3State,
    TrainingState,
)
from mindspeed.core.distributed.layerzero.zero3.flat_param import (
    FlatParameter,
    FlatParamHandle,
)


def print0(msg):
    if dist.get_rank() == 0:
        print(msg)


def _get_zero3_root_states_with_modules(
    module: nn.Module,
) -> Tuple[List[_ZeRO3State], List[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the root ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the root modules owning the states in the first
    list.

    This is similar to :func:`_get_zero3_states_with_modules` except that we
    must call :func:`_is_fsdp_root` to force a lazy initialization to determine
    the FSDP root in case lazy initialization has not yet happened.
    """
    zero3_root_states: List[_ZeRO3State] = []
    zero3_root_modules: List[nn.Module] = []
    visited_zero3_states: Set[_ZeRO3State] = set()
    # NOTE: This function assumes that `module.modules()` proceeds top-down.
    for submodule in module.modules():
        optional_state = _get_module_zero3_state(submodule)
        if (
            optional_state is not None
            and optional_state not in visited_zero3_states
            and _is_zero3_root(optional_state, submodule)
        ):
            visited_zero3_states.add(optional_state)
            zero3_root_states.append(optional_state)
            zero3_root_modules.append(submodule)
    return zero3_root_states, zero3_root_modules


def _get_zero3_root_states(module: nn.Module) -> List[_ZeRO3State]:
    """See :func:`_get_zero3_root_states_with_modules`."""
    zero3_root_states, _ = _get_zero3_root_states_with_modules(module)
    return zero3_root_states


def _is_zero3_root(state: _ZeRO3State, module: nn.Module) -> bool:
    """
    Returns if ``state`` corresponds to that of an zero3 root.

    For the wrapper code path, ``state`` and ``module`` should be the same. For
    the non-wrapper code path, ``state`` should be ``module`` 's state.
    """
    if state._is_root is None:
        raise ValueError(f"state is not initialized")
    return state._is_root


def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if div_factor > 1:
        tensor.div_(div_factor)


def _wait_for_computation_stream(
    computation_stream: torch.Stream,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the zero3 root's pre-forward to
    respect optimizer step computation.
    """
    unshard_stream.wait_stream(
        computation_stream)  # type: ignore[attr-defined]
    # Having the pre-all-gather stream wait for the current stream even if we
    # do not leverage the pre-all-gather stream is tolerable since this only
    # runs once per iteration
    # type: ignore[attr-defined]
    pre_unshard_stream.wait_stream(computation_stream)


@no_type_check
def _get_buffers_and_dtypes_for_computation(
    state: _ZeRO3State,
    root_module: nn.Module,
) -> Tuple[List[torch.Tensor], List[Optional[torch.dtype]]]:
    """
    Returns all buffers in the module tree rooted at ``root_module`` and a
    corresponding list of the buffer dtypes for computation. Each buffer dtype
    is either ``None`` if buffer mixed precision is not enabled or the buffer
    low precision dtype otherwise.
    """
    _p_assert(state._is_root, "Expects the root to cast buffers")
    buffers: List[torch.Tensor] = []
    buffer_dtypes: List[Optional[torch.dtype]] = []
    visited_buffers: Set[torch.Tensor] = set()
    # Traverse the FSDP states bottom-up so that we prefer the owning FSDP
    # instance's mixed precision setting for each buffer
    zero3_states, zero3_modules = traversal_utils._get_zero3_states_with_modules(
        root_module
    )
    for zero3_state, zero3_module in zip(reversed(zero3_states), reversed(zero3_modules)):
        for buffer_name, buffer in zero3_module.named_buffers():
            if buffer in visited_buffers:
                continue
            visited_buffers.add(buffer)
            if clean_tensor_name(buffer_name) in zero3_state._ignored_buffer_names:
                continue
            buffers.append(buffer)
            buffer_dtypes.append(zero3_state.mixed_precision.buffer_dtype)
    _p_assert(len(buffers) == len(buffer_dtypes), f"{len(buffers)} {len(buffer_dtypes)}")
    return buffers, buffer_dtypes


def _cast_buffers_to_dtype_and_device(
    buffers: List[torch.Tensor],
    buffer_dtypes: List[Optional[torch.dtype]],
    device: torch.device,
) -> None:
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
    _p_assert(
        buffer_dtypes is None or len(buffers) == len(buffer_dtypes),
        f"Expects `buffers` and `buffer_dtypes` to have the same length if "
        f"`buffer_dtypes` is specified but got {len(buffers)} and "
        f"{len(buffer_dtypes)}",
    )
    for buffer, buffer_dtype in zip(buffers, buffer_dtypes):
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)


#!===================== grad==================================================
def _reset_flat_param_grad_info_if_needed(
    handles: List[FlatParamHandle],
):
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout ZeRO3 methods, which serve
    as callsites to free the gradient memory earlier.
    """
    if not isinstance(handles, list):
        handles = [handles]
    for handle in handles:
        handle._reset_flat_param_grad_info_if_needed()


def _cast_forward_outputs(
    dtype: Optional[torch.dtype],
    output
) -> Tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
    if dtype is None:
        return output

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x
        return x.to(dtype)

    return _apply_to_tensors(cast_fn, output)
