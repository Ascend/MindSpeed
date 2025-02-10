import traceback
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
)
from torch.distributed.utils import _p_assert
from megatron.core import mpu
import mindspeed.core.distributed.layerzero.zero3._traversal_utils as traversal_utils
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _ZeRO3State,
    ZERO3_PREFIX,
    ZERO3_WRAPPED_MODULE,
    TrainingState,
)
from mindspeed.core.distributed.layerzero.zero3._init_utils import (
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    ProcessGroupType,
)

from mindspeed.core.distributed.layerzero.zero3._wrap_utils import _auto_wrap
from mindspeed.core.distributed.layerzero.zero3.api import (
    BackwardPrefetch,
    BackwardReduceScatter,
    MixedPrecision,
)
from mindspeed.core.distributed.layerzero.zero3.flat_param import FlatParameter, FlatParamHandle
from mindspeed.core.distributed.layerzero.zero3.wrap import ModuleWrapPolicy
from mindspeed.core.distributed.layerzero.runtime._forward import (
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_backward_unshard,
)
from mindspeed.core.distributed.layerzero.runtime._root_forward import _zero3_root_pre_forward
from mindspeed.core.distributed.layerzero.runtime._utils import (
    _get_zero3_root_states,
    _is_zero3_root,
    _cast_forward_outputs,
)
from mindspeed.core.distributed.layerzero.runtime._initialize import _lazy_init
from mindspeed.core.distributed.layerzero import constants


__all__ = [
    "LayerZeRO3",
]
FLAT_PARAM = "_flat_param"


class LayerZeRO3(nn.Module, _ZeRO3State):

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        tp_zero_process_group: ProcessGroupType = None,
        auto_wrap_policy: Optional[Union[Callable, ModuleWrapPolicy]] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        backward_reduce_scatter: Optional[BackwardReduceScatter] = BackwardReduceScatter.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        offload_grads: bool = False,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        forward_prefetch: bool = True,
        limit_all_gathers: bool = True,
        ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]
                     ], Optional[Iterable[torch.nn.Module]]
        ] = None,
    ):
        torch._C._log_api_usage_once("layerzero")
        super().__init__()

        _init_ignored_module_states(
            self, module, ignored_modules, ignored_states)
        _init_device_handle(self, module, self._ignored_params, device_id)
        _init_process_group_state(self, process_group)

        if auto_wrap_policy is not None:
            root_kwargs = {
                "process_group": (self.zero3_process_group, self.zero1_process_group),
                "tp_zero_process_group": tp_zero_process_group,
                "backward_prefetch": backward_prefetch,
                "backward_reduce_scatter": backward_reduce_scatter,
                "mixed_precision": mixed_precision,
                "offload_grads": offload_grads,
                "param_init_fn": param_init_fn,
                "device_id": device_id,
                "forward_prefetch": forward_prefetch,
                "limit_all_gathers": limit_all_gathers,
                "ignored_states": self._ignored_params,
            }
            _auto_wrap(
                module,
                auto_wrap_policy,
                self._ignored_modules,
                self._ignored_params,
                root_kwargs,
                LayerZeRO3,
            )

        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        _init_core_state(
            self,
            mixed_precision,
            limit_all_gathers,
            backward_prefetch_limit,
            forward_prefetch_limit,
            offload_grads,
        )
        _init_runtime_state(self)

        _init_prefetching_state(self, backward_prefetch,
                                forward_prefetch, backward_reduce_scatter)
        _init_buffer_state(self, module)
        _init_param_handle_from_module(
            self,
            module,
            device_id,
            param_init_fn,
        )
        self._zero3_wrapped_module = module

    @property
    def module(self) -> nn.Module:
        """
        Returns the wrapped module (like :class:`DistributedDataParallel`).
        """
        # FSDP's `.module` must refer to the innermost wrapped module when
        # composing with other module wrappers in order for state dict to work
        if isinstance(self._zero3_wrapped_module, ActivationWrapper):
            return getattr(self._zero3_wrapped_module, _CHECKPOINT_WRAPPED_MODULE)
        return self._zero3_wrapped_module

    @property
    def _has_params(self) -> bool:
        """Returns whether this FSDP instance manages any parameters."""
        return hasattr(self, "_handle") and self._handle is not None

    @property
    def _flat_param(self) -> Optional[FlatParameter]:
        return self._handle.flat_param if self._handle else None

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._zero3_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        if hasattr(self, ZERO3_WRAPPED_MODULE):
            # type: ignore[operator]
            return self._zero3_wrapped_module.__getitem__(key)
        return super().__getitem__(key)

    def check_is_root(self) -> bool:
        return _is_zero3_root(self, self)

    @staticmethod
    def zero3_modules(
        module: nn.Module,
        root_only: bool = False,
    ) -> List["LayerZeRO3"]:
        """
        Returns all nested ZeRO3 instances, possibly including ``module`` itself
        and only including ZeRO3 root modules if ``root_only=True``.

        Args:
            module (torch.nn.Module): Root module, which may or may not be an
                ``FSDP`` module.
            root_only (bool): Whether to return only FSDP root modules.
                (Default: ``False``)

        Returns:
            List[FullyShardedDataParallel]: FSDP modules that are nested in
            the input ``module``.
        """
        if root_only:
            return _get_zero3_root_states(module)
        return traversal_utils._get_zero3_states(module)

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """
        Returns if the user explicitly enabled buffer mixed precision.

        NOTE: Unlike parameters and gradient reduction, buffer mixed precision
        is applied at the FSDP instance level, not the ``FlatParameter`` level,
        which may be different for the composable code path.
        """
        return self.mixed_precision.buffer_dtype is not None

    def _reset_lazy_init(self) -> None:
        """
        Reset instance so :func:`_lazy_init` will run on the next forward.
        """
        self._is_root: Optional[bool] = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the forward pass for the wrapped module, inserting FSDP-specific
        pre- and post-forward sharding logic.
        """
        handle = self._handle
        with torch.autograd.profiler.record_function(
            "LayerZeRO3.forward"
        ):
            args, kwargs = _zero3_root_pre_forward(self, self, args, kwargs)
            unused = None
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_backward_unshard,
                self._zero3_wrapped_module,
                args,
                kwargs,
            )
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            with torch.autograd.profiler.record_function("Wrapped Module Forward"):
                output = self._zero3_wrapped_module(*args, **kwargs)
            output = _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )
            if constants.AUTO_CAST_OUTPUT and self._is_root:
                if mpu.is_initialized():
                    if mpu.is_pipeline_last_stage():
                        output = _cast_forward_outputs(torch.float32, output)
                else:
                    output = _cast_forward_outputs(torch.float32, output)
            return output

    def named_buffers(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Overrides :meth:`named_buffers()` to intercept buffer names and
        remove all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for buffer_name, buffer in super().named_buffers(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                buffer_name = buffer_name.replace(ZERO3_PREFIX, "")
            yield (buffer_name, buffer)

    def named_modules(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Overrides :meth:`named_buffers()` to intercept buffer names and
        remove all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for module_name, module in super().named_modules(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                module_name = module_name.replace(ZERO3_PREFIX, "")
            yield (module_name, module)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Overrides :meth:`named_parameters()` to intercept parameter names and
        remove all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        for param_name, param in super().named_parameters(*args, **kwargs):
            if should_clean_name:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(ZERO3_PREFIX, "")
            yield (param_name, param)

    def _assert_state(self, state: Union[TrainingState, List[TrainingState]]) -> None:
        """Assert we are in the given state."""
        if isinstance(state, TrainingState):
            state = [state]
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.zero3_rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager
    def no_sync(self) -> Generator:
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`no_sync()` on inner LayerZeRO instances is not supported. Please call `no_sync()` on root LayerZeRO module."
            )
        self._assert_state(TrainingState.IDLE)
        old_flags = []
        for m in self.modules():
            if isinstance(m, LayerZeRO3):
                old_flags.append((m, m._sync_gradients))
                m._sync_gradients = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                if m._sync_gradients:
                    raise ValueError(
                    "`_sync_gradients` was incorrectly set to `True` while in the `no_sync()` context manager"
                )
                m._sync_gradients = old_flag

    def zero1_parameters(self, recurse: bool = True):
        # for name, param in chain(handle. for handle in self._all_handles):
        for param in self.parameters(recurse):
            if param.requires_grad:
                yield param

    def zero_grad_buffer(self):
        '''This method is to used for accomendate with Megatron'''
        pass
