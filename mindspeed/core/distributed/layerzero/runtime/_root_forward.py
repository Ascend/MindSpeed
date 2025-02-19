from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.distributed.utils import (
    _cast_forward_inputs,
    _p_assert,
    _to_kwargs,
)
from mindspeed.core.distributed.layerzero import constants
from mindspeed.core.distributed.layerzero.zero3._common_utils import _ZeRO3State, _is_composable
from mindspeed.core.distributed.layerzero.zero3.flat_param import FlatParamHandle

from ._utils import (
    _reset_flat_param_grad_info_if_needed,
    _wait_for_computation_stream
)
from ._initialize import _lazy_init


@no_type_check
def _zero3_root_pre_forward(
    state: _ZeRO3State,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    with torch.profiler.record_function("LayerZeRO3._root_pre_forward_check"):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None,
                  "Expects a root ZeRO3 to have been set")
        if not state._is_root:
            if constants.AUTO_CAST_INPUT and _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return args, kwargs

    with torch.profiler.record_function("LayerZeRO3._root_pre_forward"):
        if state.forward_prefetch:
            handles: List[FlatParamHandle] = []
            for zero3_state in state._all_zero3_states:
                if zero3_state._handle:
                    handles.append(zero3_state._handle)
            for handle in handles:
                handle._needs_pre_forward_unshard = True

        _wait_for_computation_stream(
            state._default_stream, state._unshard_stream, state._pre_unshard_stream)
        _reset_flat_param_grad_info_if_needed(state._all_handles)

        # Prepares the forward inputs by moving them to ``compute_device``
        # the perf with/without it.
        with torch.profiler.record_function("LayerZeRO3._to_kwargs"):
            args_tuple, kwargs_tuple = _to_kwargs(
                args, kwargs, state.compute_device, False
            )
        args = args_tuple[0]
        kwargs = kwargs_tuple[0]
        return args, kwargs


@no_type_check
def _root_cast_forward_input(
    state: _ZeRO3State, module: torch.nn.Module, args, kwargs
) -> Tuple[Any, Any]:

    if module.training and state.mixed_precision is not None:
        input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
        args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
    return args, kwargs