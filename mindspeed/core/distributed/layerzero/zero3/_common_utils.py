"""
This file includes private common utilities for FSDP.
"""
import traceback
import warnings
import weakref
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)
from torch.utils._mode_utils import no_dispatch

if TYPE_CHECKING:
    from mindspeed.core.distributed.layerzero.zero3._exec_order_utils import _ExecOrderData
from mindspeed.core.distributed.layerzero.comm.hookwrap import CriticalPathEventQueue

ZERO3_WRAPPED_MODULE = "_zero3_wrapped_module"
ZERO3_PREFIX = ZERO3_WRAPPED_MODULE + "."
ZERO3_FLATTENED = "_zero3_flattened"
CRITICAL_EVENT_QUEUE = CriticalPathEventQueue()


class _DeviceHandle:
    """
    This is a simple abstraction for FSDP computing devices,
    which enables custom backends that implement CUDA-like
    semantics to be integrated with FSDP.
    """

    def __init__(self, device: torch.device, backend: Any = None):
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)
                self.__device = device
            except AttributeError as e:
                raise AttributeError(
                    f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'."
                ) from e
        else:
            self.__backend = backend

    @classmethod
    def from_device(cls, device: torch.device) -> "_DeviceHandle":
        """
        Return an device handle corresponding to the device, and through this handle,
        operations with the same semantics as CUDA can be performed on the device.
        Just return torch.cuda if the device is cuda to make attribute-access faster.
        Custom backend must first register a module with the same name with {device.type} on torch.
        """
        if device.type == "cuda":
            return cast(_DeviceHandle, torch.cuda)
        return cls(device)

    def __getattr__(self, __name: str) -> Any:
        try:
            return getattr(self.__backend, __name)
        except AttributeError as e:
            raise AttributeError(
                f"Custom backend '{self.__device.type}' not implement 'torch.{self.__device.type}.{__name}'"
            ) from e


class _UninitializedDeviceHandle:
    def __init__(self):
        pass

    def __getattribute__(self, __name: str) -> Any:
        raise RuntimeError("Trying to use an uninitialized device handle.")


class _ZeRO3State(_State):

    def __init__(self) -> None:
        self._debug_level = None
        #! zero3 related attributes
        self._ignored_modules: Set[nn.Module] = set()
        self._ignored_params: Set[nn.Parameter] = set()
        # Buffer names are cleaned (without wrapper prefixes)
        self._ignored_buffer_names: Set[str] = set()
        self.zero3_process_group: Optional[dist.ProcessGroup] = None
        #!=========================zero1 pg state===================
        self.zero1_process_group: Optional[dist.ProcessGroup] = None
        self.global_rank: int = -1
        self.world_size: int = -1
        #!==========================================================
        self.zero3_rank: int = -1
        self.zero3_world_size: int = -1
        self.limit_all_gathers: bool = False
        self.training_state = TrainingState.IDLE
        self._unshard_params_ctx: Dict[nn.Module, Generator] = {}
        self._is_root: Optional[bool] = None
        self._handle = None
        # : Dict[nn.Module, Optional[flat_param_file.FlatParamHandle]]
        self._zero3_module_to_handle = {}
        self.compute_device: Optional[torch.device] = None
        self._gradient_predivide_factor: int = 0
        # Abstract device handle for fsdp compute device. For now,
        # the compute device must implement cuda semantics used by fsdp
        self._device_handle: _DeviceHandle = _UninitializedDeviceHandle()
        # All following attributes should only be used for root states:
        # Save these static lists to avoid the repeated tree traversals
        self._all_zero3_states: List[_ZeRO3State] = []
        self._all_handles = []          # : List[flat_param_file.FlatParamHandle] = []
        self.mixed_precision = None
        self._offload_grads = False
        #!===========================streams==================================
        self._unshard_stream = None
        self._post_backward_stream = None
        self._pre_unshard_stream = None
        self._default_stream = None
        self._offload_stream = None
        self._exec_order_data: "_ExecOrderData" = None
        self._free_event_queue = None
        self._rs_event_queue = None
        self._offload_event_queue = None
        #!==========================runtime state =========================
        self.backward_prefetch = None
        self.backward_reduce_scatter = None
        self.forward_prefetch: bool = None
        self._root_pre_forward_handles: List[RemovableHandle] = []
        self._pre_forward_handles: List[RemovableHandle] = []
        self._post_forward_handles: List[RemovableHandle] = []
        self._sync_gradients: bool = False
        self._root_needs_param_sync: bool = True
        #!==========================hook state===========================
        self._post_backward_callback_queued: bool = False
        #!=================================================================

    def wait_critical_path_events(self):
        if CRITICAL_EVENT_QUEUE is None or CRITICAL_EVENT_QUEUE.empty():
            return
        with torch.profiler.record_function("LayerZeRO3: wait critical path events"):
            with CRITICAL_EVENT_QUEUE.block():
                while not CRITICAL_EVENT_QUEUE.empty():
                    event = CRITICAL_EVENT_QUEUE.pop_left()
                    if event is not None:
                        with torch.profiler.record_function(
                            "LayerZeRO3.critical_path_events"
                        ):
                            event.wait()

    @classmethod
    def record_critical_event(cls):
        if dist.get_rank() == 0:
            print("Record a critical event")
        event = torch.cuda.Event()
        event.record()
        CRITICAL_EVENT_QUEUE.enqueue(event)


def _get_module_zero3_state(module: nn.Module) -> Optional[_ZeRO3State]:
    state = _get_module_state(module)
    if state is None or not isinstance(state, _ZeRO3State):
        return None
    return state


class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """

    IDLE = auto()
    FORWARD_BACKWARD = auto()
    SUMMON_FULL_PARAMS = auto()


class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()
    SYNC_PARAMS = auto()


def _is_composable(state: _ZeRO3State):
    return not isinstance(state, nn.Module)


@no_type_check
def _module_handle(state: _ZeRO3State, module: nn.Module):
    """
    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is
    the handle that contains some parameter in ``module``.
    """
    if _is_composable(state):
        # A valid FSDP state may have no managed parameters and hence no
        # handles, meaning no entry in `_fully_sharded_module_to_handles`
        if state._handle is None:
            return None
        if module not in state._zero3_module_to_handle:
            raise AssertionError(f"Expects a fully sharded module but got {module} on rank {state.zero3_rank}")
        return state._zero3_module_to_handle[module]
    else:
        # NOTE: This assumes `module` is a `FullyShardedDataParallel` instance.
        return module._handle


@no_type_check
def _has_zero3_params(state: _ZeRO3State, module: nn.Module) -> bool:
    """Returns if ``module`` has parameters managed by LayerZeRO3."""
    return _module_handle(state, module) is not None


def clean_tensor_name(tensor_name: str) -> str:
    """
    Cleans the parameter or buffer name by removing any module wrapper
    prefixes.
    """
    tensor_name = tensor_name.replace(ZERO3_PREFIX, "")
    # it couples `CheckpointWrapper` and FSDP and also does not scale for more
    # module wrappers.
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX, "")
    return tensor_name


def _set_zero3_flattened(tensor: torch.Tensor) -> None:
    """
    Sets an attribute on ``tensor`` to mark it as flattened by FSDP. This is to
    avoid re-flattening it during nested construction.
    """
    setattr(tensor, ZERO3_FLATTENED, True)


def _is_zero3_flattened(tensor: torch.Tensor) -> bool:
    """Returns if ``tensor`` has been marked as flattened by FSDP."""
    return getattr(tensor, ZERO3_FLATTENED, False)


def _named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> List[Tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError as e:
        kwargs.pop("remove_duplicate")
        ret = list(module.named_parameters(**kwargs))
    return ret


def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    filter_fqns: Optional[List[str]] = None,
    *args,
    **kwargs,
):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.

    ``filter_fqns`` is used because some module may have its own prefix similar
    to ``FullyShardedDataParallel`` and the ``named_parameters()`` is overwritten
    to remove the prefix.
    """

    def f(module: torch.nn.Module, prefix: str, tree_level: int, *args, **kwargs):
        # Call the module function before recursing over children (pre-order)
        module_fn(module, prefix, tree_level, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is None:
                continue
            new_prefix = prefix + submodule_name + "."
            new_tree_level = tree_level + 1
            if filter_fqns is not None:
                for fqn in filter_fqns:
                    if fqn.startswith(new_prefix):
                        break
                else:
                    # DMP's named_parameter() will mess up the traversal with
                    # ``named_children`` + `named_parameter(recurse=False)``.
                    # This hack is a must to make the traversal work.
                    if (
                        submodule_name == "_zero3_wrapped_module"
                        or submodule_name == "_dmp_wrapped_module"
                    ):
                        if (
                            not torch.distributed._functional_collectives.is_torchdynamo_compiling()
                        ):
                            warnings.warn(
                                "An unexpected prefix is detected. This case "
                                " should only happen when using DMP with FSDP. "
                                f"prefix = {prefix}, "
                                f"submodule_name = {submodule_name}"
                            )
                        new_prefix = prefix
                    elif submodule_name == "module":
                        warnings.warn(
                            "An unexpected prefix is detected. This case "
                            " should only happen when DDP wraps the outer "
                            " modules while FSDP wraps the inner ones."
                            f"prefix = {prefix}, "
                            f"submodule_name = {submodule_name}"
                        )
                        new_prefix = prefix
            f(submodule, new_prefix, new_tree_level, *args, **kwargs)

    f(root_module, "", 0, *args, **kwargs)
    return return_fn(*args, **kwargs)


@no_type_check
def _assert_in_training_states(
    state: _ZeRO3State,
    training_states: List[TrainingState],
) -> None:
    """Asserts that zero3 is in the states ``_training_states``."""
    # Raise a `ValueError` instead of using `assert` to ensure that these
    # logical assertions run even if `assert`s are disabled
    if state.training_state not in training_states:
        msg = (
            f"expected to be in states {training_states} but current state is "
            f"{state.training_state}"
        )
        # Print the error on rank 0 in case this is called in the backward pass
        if state.zero3_rank == 0:
            if isinstance(state, nn.Module):
                print(f"Asserting FSDP instance is: {state}")
            print(f"ERROR: {msg}")
            traceback.print_stack()
        raise ValueError(msg)


def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None:
    if tensor.device.type not in ["cuda", torch._C._get_privateuse1_backend_name(), "npu"]:
        return

    # Don't no dispatch under torch compile like this
    with no_dispatch():
        tensor.record_stream(stream)


def _same_storage_as_data_ptr(x: torch.Tensor, data_ptr: int) -> bool:
    return x._typed_storage()._data_ptr() == data_ptr
