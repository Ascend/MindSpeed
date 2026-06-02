# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Swap-enabled optimizer for offloading parameters and optimizer states to CPU.

This module provides:

* ``SwapOptimizerMixin`` — a generic mixin that adds CPU-offloading
  (swap) support to *any* ``torch.optim.Optimizer`` subclass.  The mixin
  manages CUDA streams, pinned-CPU buffers, and async copy events.  It is
  parameterised by ``swap_state_keys`` (the optimizer-state dict keys to
  swap, e.g. ``["momentum_buffer"]`` for Muon or
  ``["exp_avg", "exp_avg_sq"]`` for Adam).

* ``SwapMuonOptimizer`` — ``SwapOptimizerMixin`` combined with
  ``OrthogonalizedOptimizer`` (Muon).  Parameters and momentum buffers
  are swapped to CPU at init time and prefetched back during ``step()``.
"""

import inspect
import types
from copy import deepcopy
from functools import wraps
from typing import Any, List

import torch

from mindspeed.core.optimizer.muon.orthogonalized_optimizer import OrthogonalizedOptimizer
from mindspeed.core.optimizer.muon.orthogonalized_optimizer import _fp32_matmul_precision
from mindspeed.args_utils import get_full_args as get_args


# ======================================================================
# Generic swap mixin
# ======================================================================


class SwapOptimizerMixin:
    """Mixin that adds CPU-offloading swap support to an optimizer.

    Subclasses must:

    * Set ``swap_state_keys`` to the list of optimizer-state dict keys
      that should be swapped (e.g. ``["momentum_buffer"]``).
    * Call ``cls._swap_init(optimizer)`` after the base optimizer has been fully
      initialised (i.e. after all param groups and states exist).
    * Override ``step()`` to call the three high-level swap operations
      around the actual optimizer update.

    The mixin does **not** depend on Megatron's DistributedOptimizer.
    """

    swap_state_keys: List[str] = []

    # -- class-level swap infrastructure (shared across instances) --------
    _swap_to_device_stream = None
    _swap_to_host_stream = None
    _swap_numel: int = 0

    # per-param CPU mirrors:  param -> { "param": cpu_tensor, <state_key>: cpu_tensor, ... }
    _param_to_cpu_states: dict = {}
    # per-param optimizer state ref (for classmethod access)
    _state_map: dict = {}
    # per-param event tracking
    _swap_to_device_events: dict = {}
    _swap_to_host_events: dict = {}
    _copy_to_model_events: dict = {}
    # main-param -> model-param mapping
    _main_param_to_model_param: dict = {}

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_streams(cls):
        if cls._swap_to_device_stream is None:
            cls._swap_to_device_stream = torch.cuda.Stream()
        if cls._swap_to_host_stream is None:
            cls._swap_to_host_stream = torch.cuda.Stream()

    # ------------------------------------------------------------------
    # Init-time: create CPU mirrors and swap everything to host
    # ------------------------------------------------------------------

    @classmethod
    def swap_init(cls, optimizer):
        """Create pinned-CPU mirrors for all params + states and swap to host.

        Must be called after the base optimizer is fully initialized (all
        param groups populated and optimizer states allocated).
        """
        cls._ensure_streams()

        model_data, main_data = [], []
        for model_group, main_group in zip(optimizer.float16_groups, optimizer.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param)
                main_data.append(main_param)

        for model_param, main_param in zip(model_data, main_data):
            if main_param in cls._param_to_cpu_states or not model_param.requires_grad:
                continue

            state = optimizer.state[main_param]

            # CPU mirror for the param itself
            cpu_states = {"param": torch.empty_like(main_param, pin_memory=True, device="cpu")}
            cpu_states["param"].copy_(main_param, non_blocking=True)

            # CPU mirrors for each optimizer state key
            for key in cls.swap_state_keys:
                if key not in state:
                    state[key] = torch.zeros_like(main_param.data)

                tensor = state[key]
                if tensor is None:
                    cpu_states[key] = None
                    continue
                cpu_tensor = torch.empty_like(tensor, pin_memory=True, device="cpu")
                cpu_tensor.copy_(tensor, non_blocking=True)
                cpu_states[key] = cpu_tensor

            cls._param_to_cpu_states[main_param] = cpu_states
            cls._state_map[main_param] = state
            cls._main_param_to_model_param[main_param] = model_param
            cls._swap_to_host_events[main_param] = None
            cls._copy_to_model_events[main_param] = None

            # Free GPU storage for param
            main_param.storage().resize_(0)
            # Free GPU storage for each state tensor
            for key in cls.swap_state_keys:
                if key in state and state[key] is not None:
                    state[key].storage().resize_(0)

        # Compute swap batch size
        if cls._swap_numel == 0:
            args = get_args()
            swap_times = getattr(args, "swap_optimizer_times", 16)
            total = sum(p.numel() for p in main_data)
            cls._swap_numel = max(1, total // swap_times)

    # ------------------------------------------------------------------
    # Low-level swap primitives
    # ------------------------------------------------------------------

    @classmethod
    def _swap_tensors_to_device(cls, param):
        """Async copy param + state tensors from CPU to GPU."""
        cpu = cls._param_to_cpu_states.get(param)
        if cpu is None:
            return

        if param.storage().size() == 0:
            param.storage().resize_(cpu["param"].storage().size())
            param.copy_(cpu["param"], non_blocking=True)

        state = cls._state_map.get(param)
        if state is not None:
            for key in cls.swap_state_keys:
                t = state.get(key)
                if t is None or t.storage().size() != 0:
                    continue
                t.storage().resize_(cpu[key].storage().size())
                t.copy_(cpu[key], non_blocking=True)

        cls._swap_to_device_events[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def _wait_swap_to_device(cls, param):
        event = cls._swap_to_device_events.get(param)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            cls._swap_to_device_events[param] = None

    @classmethod
    def _swap_tensors_to_host(cls, param):
        """Async copy param + state tensors from GPU to CPU."""
        cpu = cls._param_to_cpu_states.get(param)
        if cpu is None:
            return

        if param.storage().size() != 0:
            cpu["param"].copy_(param, non_blocking=True)
            param.storage().resize_(0)

        state = cls._state_map.get(param)
        if state is not None:
            for key in cls.swap_state_keys:
                t = state.get(key)
                if t is None or t.storage().size() == 0:
                    continue
                cpu[key].copy_(t, non_blocking=True)
                t.storage().resize_(0)

        cls._swap_to_host_events[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def _copy_param_to_model(cls, param):
        model_param = cls._main_param_to_model_param.get(param)
        if model_param is not None and model_param is not param:
            model_param.data.copy_(param)
        cls._copy_to_model_events[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def _wait_copy_to_model(cls, param):
        event = cls._copy_to_model_events.get(param)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            cls._copy_to_model_events[param] = None

    # ------------------------------------------------------------------
    # High-level swap operations (used inside step)
    # ------------------------------------------------------------------

    @classmethod
    def swap_prefetch_to_device(cls, params_list, idx, swap_count):
        """Async prefetch a batch of params from CPU to GPU.

        Returns (new_idx, new_swap_count).
        """
        torch.cuda.current_stream().wait_stream(cls._swap_to_host_stream)
        with torch.cuda.stream(cls._swap_to_device_stream):
            torch.cuda.current_stream().wait_stream(cls._swap_to_host_stream)
            while idx < len(params_list) and (
                swap_count + params_list[idx].numel() <= cls._swap_numel or swap_count <= 0
            ):
                cls._swap_tensors_to_device(params_list[idx])
                swap_count += params_list[idx].numel()
                idx += 1
        return idx, swap_count

    @classmethod
    def swap_wait_device_ready(cls, param):
        """Block until *param* has been fully swapped to GPU."""
        cls._wait_swap_to_device(param)

    @classmethod
    def swap_copy_back_and_release(cls, param):
        """Copy updated param back to model param, then async swap to CPU."""
        cls._copy_param_to_model(param)
        with torch.cuda.stream(cls._swap_to_host_stream):
            cls._wait_copy_to_model(param)
            cls._swap_tensors_to_host(param)


def swap_layer_wise_distributed_optimizer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args: Any, **kwargs: Any):
        fn(self, *args, **kwargs)
        SwapOptimizerMixin.swap_state_keys = ["momentum_buffer"]
        for optimizer in self.chained_optimizers:
            if isinstance(optimizer.optimizer, OrthogonalizedOptimizer):
                SwapOptimizerMixin.swap_init(optimizer)
                optimizer._copy_main_params_to_model_params = types.MethodType(dummy_function, optimizer)
                optimizer._copy_model_params_to_main_params = types.MethodType(
                    _copy_model_params_to_main_params_with_swap, optimizer
                )
                optimizer.state_dict = types.MethodType(state_dict_swap_wrapper(optimizer.state_dict), optimizer)
                optimizer.load_state_dict = types.MethodType(
                    load_state_dict_swap_wrapper(optimizer.load_state_dict), optimizer
                )

    return wrapper


def dummy_function(*args: Any, **kwargs: Any):
    pass


def _copy_model_params_to_main_params_with_swap(self):
    for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
        for model_param, main_param in zip(model_group, main_group):
            # Swap in
            cpu_states = SwapOptimizerMixin._param_to_cpu_states.get(main_param)
            if cpu_states is not None and main_param.storage().size() == 0:
                SwapOptimizerMixin._swap_tensors_to_device(main_param)
                SwapOptimizerMixin._wait_swap_to_device(main_param)

            # copy
            main_param.data.copy_(model_param.data)

            # Swap out
            if cpu_states is not None:
                SwapOptimizerMixin._swap_tensors_to_host(main_param)

    torch.cuda.synchronize()  # wait swap out events


def state_dict_swap_wrapper(fn):
    """Wrap an optimizer's ``state_dict()`` so that swapped-out tensors
    are temporarily swapped back to device before the state is read,
    then swapped out again afterwards.

    Usage::

        optimizer.state_dict = state_dict_swap_wrapper(optimizer.state_dict)
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Swap in all tracked params to device so state_dict can read them
        swapped_params = []
        for param in SwapOptimizerMixin._param_to_cpu_states:
            if param.storage().size() == 0:
                SwapOptimizerMixin._swap_tensors_to_device(param)
                swapped_params.append(param)

        # Wait for all swap-in operations to complete
        for param in swapped_params:
            SwapOptimizerMixin._wait_swap_to_device(param)

        # Call the original state_dict
        result = fn(self, *args, **kwargs)
        result = deepcopy(result)

        # Swap out: release back to CPU
        for param in swapped_params:
            SwapOptimizerMixin._swap_tensors_to_host(param)

        torch.cuda.synchronize()
        return result

    return wrapper


def load_state_dict_swap_wrapper(fn):
    """Wrap an optimizer's ``load_state_dict()`` so that swapped-out
    tensors are temporarily swapped back to device before the state is
    loaded, then swapped out again afterwards (which also updates the
    CPU mirrors with the newly loaded values).

    Usage::

        optimizer.load_state_dict = load_state_dict_swap_wrapper(optimizer.load_state_dict)
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Swap in all tracked params to device so load_state_dict can write into them
        swapped_params = []
        for param in SwapOptimizerMixin._param_to_cpu_states:
            if param.storage().size() == 0:
                SwapOptimizerMixin._swap_tensors_to_device(param)
                swapped_params.append(param)

        # Wait for all swap-in operations to complete
        for param in swapped_params:
            SwapOptimizerMixin._wait_swap_to_device(param)

        # Call the original load_state_dict
        if hasattr(fn, "__self__") or inspect.ismethod(fn):
            result = fn(*args, **kwargs)
        else:
            result = fn(self, *args, **kwargs)

        # Swap out: copy updated values back to CPU mirrors and free device storage
        for param in swapped_params:
            SwapOptimizerMixin._swap_tensors_to_host(param)

        torch.cuda.synchronize()
        return result

    return wrapper


@torch.no_grad()
def swap_muon_step(self, closure=None):
    if closure is not None:
        loss = closure()
    else:
        loss = None

    # Collect params that need updating
    swap_count = 0
    idx = 0

    params_list = []
    groups_list = []
    for group in self.param_groups:
        for param in group["params"]:
            params_list.append(param)
            groups_list.append(group)

    for param, group in zip(params_list, groups_list):
        # Prefetch to device
        if swap_count <= 0:
            idx, swap_count = SwapOptimizerMixin.swap_prefetch_to_device(params_list, idx, swap_count)
        SwapOptimizerMixin.swap_wait_device_ready(param)

        # Muon update
        grad = param.grad
        state = self.state[param]
        self._apply_weight_decay_inplace(param, grad, group["lr"], group["weight_decay"])

        # update momentum buffer with EMA of gradient
        state["momentum_buffer"].lerp_(grad, 1.0 - group["momentum"])
        # include nesterov momentum
        if self.nesterov:
            grad = grad.lerp(state["momentum_buffer"], group["momentum"])
        else:
            grad = state["momentum_buffer"]

        with _fp32_matmul_precision(self.fp32_matmul_prec):
            group_kwargs = {key: value for key, value in group.items() if key != "params"}
            orth_grad = self.orthogonalize(param, grad, **group_kwargs)

        # perform weight update with pre and post weight update functions for subclass customization
        self.pre_weight_update_fn_inplace(param, orth_grad)
        param.add_(orth_grad, alpha=-group["lr"])
        self.post_weight_update_fn_inplace(param)

        # Copy back and release to host
        SwapOptimizerMixin.swap_copy_back_and_release(param)
        swap_count -= param.numel()

    return loss
