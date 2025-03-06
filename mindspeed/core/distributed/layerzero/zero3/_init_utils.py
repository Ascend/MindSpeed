import collections
import warnings
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist

import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from megatron.core import mpu

from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _DeviceHandle,
    _ZeRO3State,
    _get_module_zero3_state,
    _is_zero3_flattened,
    _named_parameters_with_duplicates,
    clean_tensor_name,
    TrainingState,
)
from mindspeed.core.distributed.layerzero.zero3.api import (
    BackwardPrefetch,
    BackwardReduceScatter,
    MixedPrecision,
)
from mindspeed.core.distributed.layerzero.zero3.flat_param import (
    FlatParameter,
    FlatParamHandle,
)
from mindspeed.core.distributed.layerzero.zero3._limiter import _FreeEventQueue
import mindspeed.core.distributed.layerzero.zero3._exec_order_utils as exec_order_utils
import mindspeed.core.distributed.layerzero.zero3._traversal_utils as traversal_utils
import mindspeed.core.distributed.layerzero.zero3.fsdp as zero3_file


PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)
ZERO3_SYNCED = "_zero3_synced"
# Overall specification of process group.
ProcessGroupType = Tuple[dist.ProcessGroup, dist.ProcessGroup]


def _get_gradient_predivide_factor(world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
        factor *= 2
    return float(factor)


@no_type_check
def _init_process_group_state(
    state: _ZeRO3State,
    process_group: ProcessGroupType,
) -> _ZeRO3State:

    state.zero3_process_group, state.zero1_process_group = process_group
    state.zero3_rank = state.zero3_process_group.rank()
    data_parallel_world_size = state.zero1_process_group.size()
    state.world_size = data_parallel_world_size
    state.global_rank = dist.get_rank()
    if mpu.is_initialized():
        state._gradient_predivide_factor = float(dist.get_world_size(
            mpu.get_data_parallel_group(with_context_parallel=True)))
    else:
        state._gradient_predivide_factor = data_parallel_world_size
    return state


@no_type_check
def _init_ignored_module_states(
    state: _ZeRO3State,
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]],
    ignored_states: Union[
        Optional[Iterable[torch.nn.Parameter]
                 ], Optional[Iterable[torch.nn.Module]]
    ] = None,
) -> _ZeRO3State:
    if ignored_modules is not None and ignored_states is not None:
        raise ValueError(
            "Cannot pass both ignored_modules and ignored_states at the "
            "same time. Please just pass ignored_states."
        )
    ignored_parameters = None
    passed_as_ignored_states = ignored_states is not None
    if passed_as_ignored_states:
        ignored_states_list = list(ignored_states)
        _check_ignored_states(ignored_states_list, True)
    else:
        ignored_states_list = []
        _check_ignored_states(
            list(ignored_modules) if ignored_modules is not None else [], False
        )
    if len(ignored_states_list) > 0:
        if isinstance(ignored_states_list[0], nn.Parameter):
            ignored_parameters = ignored_states_list
        else:
            ignored_modules = ignored_states_list
    state._ignored_modules = _get_ignored_modules(module, ignored_modules)
    state._ignored_params = _get_ignored_params(
        module,
        state._ignored_modules,
        ignored_parameters,
    )
    state._ignored_buffer_names = _get_ignored_buffer_names(
        module,
        state._ignored_modules,
    )
    return state


def _check_ignored_states(
    ignored_states: List[Any], passed_as_ignored_states: bool
) -> None:
    """
    Checks that the ignored states are uniformly parameters or uniformly
    modules. We may remove this check in the future if we permit mixing.
    """
    if len(ignored_states) == 0:
        return
    if passed_as_ignored_states:
        all_params = all(isinstance(state, nn.Parameter)
                         for state in ignored_states)
        all_modules = all(isinstance(state, nn.Module)
                          for state in ignored_states)
        if not all_params and not all_modules:
            # Sort for consistent ordering for unit test regex matching
            sorted_types = sorted(
                {type(state) for state in ignored_states}, key=lambda x: repr(x)
            )
            raise ValueError(
                "ignored_states expects all nn.Parameter or all nn.Module list "
                f"elements but got types {sorted_types}"
            )
    else:
        if not all(isinstance(state, nn.Module) for state in ignored_states):
            sorted_types = sorted(
                {type(state) for state in ignored_states}, key=lambda x: repr(x)
            )
            raise ValueError(
                "ignored_modules expects nn.Module list elements but got "
                f"types {sorted_types}"
            )


@no_type_check
def _init_device_handle(
    state: _ZeRO3State,
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_id: Optional[Union[int, torch.device]],
) -> _ZeRO3State:
    determined_device = None
    if device_id is not None:
        determined_device = (
            device_id
            if isinstance(device_id, torch.device)
            else torch.device(device_id)
        )
    if determined_device is None:
        for param in _get_orig_params(module, ignored_params):
            if param.device.type in {"cpu", "meta"}:
                continue
            if determined_device is None:
                determined_device = param.device
            else:
                if param.device.type != determined_device.type:
                    raise RuntimeError(
                        f"FSDP does not support modules with different device types "
                        f"but got params on {determined_device.type} and {param.device.type}"
                    )
        determined_device = determined_device or torch.device(
            "cuda", torch.cuda.current_device()
        )

    state._device_handle = _DeviceHandle.from_device(determined_device)
    return state


@no_type_check
def _init_buffer_state(
    state: _ZeRO3State,
    module: nn.Module,
) -> _ZeRO3State:
    state._buffer_names = _get_buffer_names(module)
    # Save a mapping from clean fully-qualified buffer name (starting from
    # `module`) to its original dtype for restoring that dtype during model
    # checkpointing when buffer mixed precision is enabled. The names should
    # be clean since the casting happens in a `summon_full_params()` context.
    _buffer_name_to_orig_dtype: Dict[str, torch.dtype] = {}
    for buffer_name, buffer in module.named_buffers():
        buffer_name = clean_tensor_name(buffer_name)
        _buffer_name_to_orig_dtype[buffer_name] = buffer.dtype
    state._buffer_name_to_orig_dtype = _buffer_name_to_orig_dtype
    return state


@no_type_check
def _init_core_state(
    state: _ZeRO3State,
    mixed_precision: Optional[MixedPrecision],
    limit_all_gathers: bool,
    backward_prefetch_limit: int,
    forward_prefetch_limit: int,
    offload_grads: bool = False
) -> _ZeRO3State:
    # We clamp the strategy to `NO_SHARD` for world size of 1 since they are
    # currently functionally equivalent. This may change if/when we integrate
    # FSDP with MoE.
    state.mixed_precision = mixed_precision or MixedPrecision()
    if mixed_precision is not None:
        torch._C._log_api_usage_once(
            f"mixed_precision.{str(state.mixed_precision)}"
        )

    state.limit_all_gathers = limit_all_gathers
    state.training_state = TrainingState.IDLE
    state._is_root = None
    state._free_event_queue = _FreeEventQueue()
    state._rs_event_queue = _FreeEventQueue()
    state._offload_event_queue = _FreeEventQueue()
    state._offload_grads = offload_grads
    # ==========================================
    state._debug_level = dist.get_debug_level()
    state._exec_order_data = exec_order_utils._ExecOrderData(
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    #! add support for zero1 events
    # Mapping from fully sharded module to the handles it is responsible to
    # unshard and reshard (see [Note: Fully Sharded Module])
    _fully_sharded_module_to_handle: Dict[nn.Module, FlatParamHandle] = dict()
    state._zero3_module_to_handle = _fully_sharded_module_to_handle
    # Invariant: `state.params` contains exactly the `FlatParameter`s of the
    # handles in `state._handle`
    _handle: FlatParamHandle = None
    state._handle = _handle
    params: List[FlatParameter] = []
    state.params = params
    return state


@no_type_check
def _init_runtime_state(
    state: _ZeRO3State,
) -> _ZeRO3State:
    _root_pre_forward_handles: List[RemovableHandle] = []
    state._root_pre_forward_handles = _root_pre_forward_handles
    _pre_forward_handles: List[RemovableHandle] = []
    state._pre_forward_handles = _pre_forward_handles
    _post_forward_handles: List[RemovableHandle] = []
    state._post_forward_handles = _post_forward_handles
    state._sync_gradients = True
    # Used to prevent running the pre-backward hook multiple times
    return state


@no_type_check
def _init_prefetching_state(
    state: _ZeRO3State,
    backward_prefetch: BackwardPrefetch,
    forward_prefetch: bool,
    backward_reduce_scatter: BackwardReduceScatter
) -> _ZeRO3State:
    state.backward_prefetch = backward_prefetch
    state.forward_prefetch = forward_prefetch
    state.backward_reduce_scatter = backward_reduce_scatter
    # The data structures use tuples of handles to generalize over the case
    # where a module's forward involves multiple handles.
    return state


@no_type_check
def _init_param_handle_from_module(
    state: _ZeRO3State,
    zero3_module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
) -> _ZeRO3State:
    """
    Initializes a ``FlatParamHandle`` from a module ``fully_sharded_module``.
    """
    _check_single_device_module(zero3_module, state._ignored_params, device_id)
    device_from_device_id = _get_device_from_device_id(
        device_id, state.global_rank)
    _move_module_to_device(
        zero3_module, state._ignored_params, device_from_device_id
    )
    state.compute_device = _get_compute_device(
        zero3_module,
        state._ignored_params,
        device_from_device_id,
        state.global_rank,
    )

    managed_params = list(_get_orig_params(
        zero3_module, state._ignored_params))
    for param in managed_params:
        if len(param.shape) == 1:
            param._is_1D_param = True
    _init_param_handle_from_params(
        state, managed_params, zero3_module)
    return state


@no_type_check
def _init_param_handle_from_params(
    state: _ZeRO3State,
    params: List[nn.Parameter],
    zero3_module: nn.Module,
):
    if len(params) == 0:
        return
    handle = FlatParamHandle(
        params,
        zero3_module,
        state.compute_device,
        state.mixed_precision.param_dtype,
        state.mixed_precision.reduce_dtype,
        state.zero3_process_group,
        state.zero1_process_group,
        state._offload_grads
    )
    handle.shard()
    if state._handle is not None:
        raise ValueError(f"state handle has been initialized")
    state.params.append(handle.flat_param)
    state._handle = handle
    state._zero3_module_to_handle[handle._zero3_module] = handle


def _get_ignored_modules(
    root_module: nn.Module,
    _ignored_modules: Optional[Iterable[torch.nn.Module]],
) -> Set[nn.Module]:
    """
    Checks that ``_ignored_modules`` is an iterable of ``nn.Module`` s without
    any FSDP instances, and returns the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.

    ``_ignored_modules`` represents the argument passed by the user to FSDP.
    """
    msg_prefix = "`ignored_modules` should be an iterable of `torch.nn.Module`s "
    try:
        ignored_root_modules = (
            set(_ignored_modules) if _ignored_modules is not None else set()
        )
    except TypeError as e:
        raise TypeError(
            msg_prefix + f"but got {type(_ignored_modules)}") from e
    for module in ignored_root_modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                msg_prefix + f"but got an iterable with {type(module)}")
        if _get_module_zero3_state(module):
            raise ValueError(
                "`ignored_modules` should not include FSDP modules")
    # Treat modules that cannot compose with `fully_shard` as ignored modules,
    # meaning that their subtrees are ignored
    for module in root_module.modules():
        if not traversal_utils._composable(module):
            ignored_root_modules.add(module)
    # NOTE: Even if `ignored_root_modules` is empty, do not return early so
    # that this FSDP instance can get any ignored modules from its children.

    # Include child modules and exclude nested FSDP modules themselves
    ignored_modules = {
        child
        for module in ignored_root_modules
        for child in module.modules()
        if not isinstance(child, zero3_file.LayerZeRO3)
    }
    if root_module in ignored_modules:
        warnings.warn(
            "Trying to ignore the top-level module passed into the FSDP "
            "constructor itself will result in all parameters being "
            f"ignored and is not well-supported: {module}"
        )
    # Include nested FSDP modules' ignored modules
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_zero3_state(submodule)
        if optional_fsdp_state is not None:
            if not hasattr(optional_fsdp_state, "_ignored_modules"):
                raise AttributeError(
                    "State has not attribute _ignored_modules")
            ignored_modules.update(optional_fsdp_state._ignored_modules)
    return ignored_modules


def _get_ignored_params(
    root_module: torch.nn.Module,
    ignored_modules: Set[torch.nn.Module],
    ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
) -> Set[torch.nn.Parameter]:
    """
    Returns the parameters of the modules in ``ignored_modules`` and
    the parameters in ``ignored_parameters``, excluding any :class:`FlatParameter` s.
    """
    all_ignored_params: Set[torch.nn.Parameter] = set()

    params_in_ignored_modules = {
        p for m in ignored_modules for p in m.parameters() if not _is_zero3_flattened(p)
    }

    all_ignored_params.update(params_in_ignored_modules)

    if ignored_parameters is not None:
        params_in_ignored_parameters = {
            p for p in ignored_parameters if not _is_zero3_flattened(p)
        }
        all_ignored_params.update(params_in_ignored_parameters)

    # Always include nested FSDP modules' ignored parameters
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_zero3_state(submodule)
        if optional_fsdp_state is not None:
            if not hasattr(optional_fsdp_state, "_ignored_params"):
                raise AttributeError("State has not attribute _ignored_params")
            all_ignored_params.update(optional_fsdp_state._ignored_params)

    return all_ignored_params


def _get_ignored_buffer_names(
    root_module: torch.nn.Module,
    ignored_modules: Set[torch.nn.Module],
) -> Set[str]:
    """
    Returns the cleaned buffer FQNs in ``ignored_modules``
    """
    all_ignored_buffer_names: Set[str] = set()

    buffers_in_ignored_modules = {
        buffer for m in ignored_modules for buffer in m.buffers()
    }

    all_ignored_buffer_names.update(
        {
            clean_tensor_name(buffer_name)
            for buffer_name, buffer in root_module.named_buffers()
            if buffer in buffers_in_ignored_modules
        }
    )

    # Always include nested FSDP modules' ignored buffer names
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_zero3_state(submodule)
        if optional_fsdp_state is not None:
            if not hasattr(optional_fsdp_state, "_ignored_buffer_names"):
                raise AttributeError(
                    "State has not attribute _ignored_buffer_names")
            all_ignored_buffer_names.update(
                optional_fsdp_state._ignored_buffer_names)

    return all_ignored_buffer_names


def _get_buffer_names(root_module: nn.Module) -> Set[str]:
    """
    Returns the fully prefixed names of all buffers in the module hierarchy
    rooted at ``root_module`` as a class:`set`.
    """
    return {
        clean_tensor_name(buffer_name) for buffer_name, _ in root_module.named_buffers()
    }


def _check_single_device_module(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_id: Optional[Union[int, torch.device]],
) -> None:
    """
    Raises an error if ``module`` has original parameters on multiple devices,
    ignoring the parameters in ``ignored_params``. Thus, after this method, the
    module must be either fully on the CPU or fully on a non-CPU device.
    """
    devices = {param.device for param in _get_orig_params(
        module, ignored_params)}
 
    if len(devices) == 2 and torch.device("cpu") in devices:
        if device_id is None:
            raise RuntimeError(
                "To support a module with both CPU and GPU params, "
                "please pass in device_id argument."
            )
    elif len(devices) > 1:
        raise RuntimeError(
            f"ZeRO3 only supports single device modules but got params on {devices}"
        )


def _get_device_from_device_id(
    device_id: Optional[Union[int, torch.device]],
    rank: int,
) -> Optional[torch.device]:
    """
    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    if device_id is None:
        return None
    device = (
        device_id if isinstance(
            device_id, torch.device) else torch.device(device_id)
    )
    return device


def _move_module_to_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
) -> None:
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        # BFS from `module` without traversing any nested FSDP instances to
        # collect the parameters/buffers that have not yet been managed
        queue: Deque[nn.Module] = collections.deque()
        queue.append(module)
        params: List[nn.Parameter] = []
        buffers: List[torch.Tensor] = []
        while queue:
            curr_module = queue.popleft()
            params.extend(
                param
                for param in curr_module.parameters(recurse=False)
                if param.device == cpu_device
            )
            buffers.extend(
                buffer
                for buffer in curr_module.buffers(recurse=False)
                if buffer.device == cpu_device
            )
            for submodule in curr_module.children():
                if not isinstance(submodule, zero3_file.LayerZeRO3):
                    queue.append(submodule)

        _move_states_to_device(params, buffers, device_from_device_id)
        return
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device == cpu_device:
        _warn_cpu_init()


def _move_states_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    Precondition: ``_check_single_device_module()`` and module's parameters and
    buffers have been materialized if needed.
    """
    if len(params) == 0 and len(buffers) == 0:
        return
    if len(params) > 0:
        current_device = params[0].device
    elif len(buffers) > 0:
        current_device = buffers[0].device
    cpu_device = torch.device("cpu")
    if device_from_device_id is not None:
        # Move the parameters and buffers like the `.data` code path in
        # `nn.Module._apply()`, which underlies `nn.Module.to()`
        for param in params:
            with torch.no_grad():
                param.data = param.to(device_from_device_id)
                if param.grad is not None:
                    param.grad.data = param.grad.to(device_from_device_id)
        for buffer in buffers:
            buffer.data = buffer.to(device_from_device_id)
    elif current_device == cpu_device:
        _warn_cpu_init()


def _warn_cpu_init():
    warnings.warn(
        "The passed-in `module` is on CPU and will thus have FSDP's sharding "
        "initialization run on CPU, which may be slower than on GPU. We "
        "recommend passing in the `device_id` argument for FSDP to move "
        "`module` to GPU for the sharding initialization. `module` must also "
        "be on GPU device to work with the `sync_module_states=True` flag "
        "since that requires GPU communication."
    )


def _get_compute_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
    rank: int,
) -> torch.device:
    """
    Determines and returns this FSDP instance's compute device. If a device is
    specified by ``device_id``, then returns that device. Otherwise, If the
    module is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current
    device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA GPU device with its explicit index.

    Precondition: ``_check_single_device_module()`` and
    ``_move_module_to_device()``.
    """
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type != "cpu":
        compute_device = param.device
    else:
        if device_from_device_id is not None and device_from_device_id.type != "cuda":
            compute_device = device_from_device_id
        else:
            compute_device = torch.device("cuda", torch.cuda.current_device())
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(
            f"Inconsistent compute device and `device_id` on rank {rank}: "
            f"{compute_device} vs {device_from_device_id}"
        )
    return compute_device


def _get_orig_params(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Iterator[nn.Parameter]:
    param_gen = module.parameters()
    try:
        while True:
            param = next(param_gen)
            if param not in ignored_params and not _is_zero3_flattened(param):
                yield param
    except StopIteration:
        pass


def _check_orig_params_flattened(
    zero3_module,
    ignored_params: Set[nn.Parameter],
) -> None:
    """
    Checks that all original parameters have been flattened and hence made
    invisible to ``named_parameters()`` for the module hierarchy rooted at
    ``zero3_module``. This should be called as a sanity check after flattening
    the wrapped module's parameters.
    """
    for param_name, param in _named_parameters_with_duplicates(zero3_module):
        if param not in ignored_params and not _is_zero3_flattened(param):
            raise RuntimeError(
                f"Found an unflattened parameter: {param_name}; "
                f"{param.size()} {param.__class__}"
            )
