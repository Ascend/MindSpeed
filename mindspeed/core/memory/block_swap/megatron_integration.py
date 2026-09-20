# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""Megatron optimizer integration: automatic materialize/refresh/release
around the optimizer update window.

Adopting a tensor installs class-level wrappers on the Megatron optimizers,
so the standard Megatron + MindSpeed training loop needs no manual steps:

- ``Float16OptimizerWithFloat16Params.step_with_ready_grads`` /
  ``DistributedOptimizer.step_with_ready_grads``: every execution path
  (``ChainedOptimizer.step`` / direct ``.step()``) funnels into them, and
  they contain the only main->model parameter copies
  (``_copy_main_params_to_model_params``). The wrapper materializes all
  managed tensors before the update and refreshes their mirrors (D2H) +
  releases afterwards - without this, the update would either write released
  storage or leave stale mirrors (the next backward would re-materialize old
  weights and compute wrong gradients).
- ``MixedPrecisionOptimizer.reload_model_params`` /
  ``ChainedOptimizer.reload_model_params``: reload reads model param values,
  so the tensors are materialized around it (no mirror refresh: model params
  are not modified).

The wrappers wrap the *current* symbol of each class (they may already carry
MindSpeed feature patches), are idempotent through a function marker, and are
re-applied if a later repatch resets the chain. Megatron is imported lazily;
when absent the API stays usable for pure-torch inference.
"""

from functools import wraps

from mindspeed.core.memory.block_swap.swap_state import get_swap_manager

_WRAPPER_MARKER = '_block_swap_wrapped'


def _wrap_step_with_ready_grads(step_func):
    @wraps(step_func)
    def step_with_swap(self, *args, **kwargs):
        manager = get_swap_manager()
        if not manager.states:
            return step_func(self, *args, **kwargs)
        manager.swap_all_in()
        try:
            result = step_func(self, *args, **kwargs)
        finally:
            # tensors were updated on device: refresh mirrors (D2H) + release
            manager.swap_all_out(copy_to_host=True)
        return result

    return step_with_swap


def _wrap_reload_model_params(reload_func):
    @wraps(reload_func)
    def reload_with_swap(self, *args, **kwargs):
        manager = get_swap_manager()
        if not manager.states:
            return reload_func(self, *args, **kwargs)
        manager.swap_all_in()
        try:
            result = reload_func(self, *args, **kwargs)
        finally:
            manager.swap_all_out(copy_to_host=False)
        return result

    return reload_with_swap


def _patch_method(cls, name, wrap_func):
    current = getattr(cls, name, None)
    if current is None or getattr(current, _WRAPPER_MARKER, False):
        return
    wrapper = wrap_func(current)
    setattr(wrapper, _WRAPPER_MARKER, True)
    setattr(cls, name, wrapper)


def ensure_megatron_optimizer_integration():
    """Idempotently install the Megatron optimizer wrappers (lazy import)."""
    try:
        import megatron  # noqa: F401
    except ImportError:
        return  # Megatron absent: pure-torch usage stays available
    # Megatron is present: a failing submodule import here would silently
    # disable the integration, so let it raise loudly instead
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        Float16OptimizerWithFloat16Params,
        MixedPrecisionOptimizer,
    )

    for cls in (Float16OptimizerWithFloat16Params, DistributedOptimizer):
        _patch_method(cls, 'step_with_ready_grads', _wrap_step_with_ready_grads)
    for cls in (MixedPrecisionOptimizer, ChainedOptimizer):
        _patch_method(cls, 'reload_model_params', _wrap_reload_model_params)
