import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

import torch.nn as nn
from torch.distributed.fsdp._wrap_utils import _override_module_mixed_precision, _validate_frozen_params, _warn_on_overridden_mixed_precision
from megatron.training.global_vars import get_args
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
)
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    _get_module_zero3_state,
)
from mindspeed.core.distributed.layerzero.zero3.wrap import (
    _construct_wrap_fn,
    _or_policy,
    _Policy,
    _post_order_apply,
    _recursive_wrap,
    _run_mixed_precision_override_policy,
    _run_tensor_parallel_pg_override_policy,
    _wrap_module_cls_individually,
)


def _auto_wrap(
    root_module: nn.Module,
    policy: Union[Callable, _Policy],
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    root_kwargs: Dict[str, Any],
    fsdp_fn: Callable,  # e.g. `FullyShardedDataParallel` or `fully_shard`
):
    """
    Auto wraps modules in ``root_module`` 's tree according to ``policy``
    following a post-order traversal.

    Precondition: ``root_kwargs`` should contain all arguments except
    ``module``. This function accepts the kwargs dict directly since it gets
    forwarded into the post-order traversal function.
    """
    mixed_precision = root_kwargs["mixed_precision"]
    is_wrapper = inspect.isclass(fsdp_fn)
    _check_nested_wrapping(root_module)

    if isinstance(policy, _Policy):
        root_kwargs["auto_wrap_policy" if is_wrapper else "policy"] = None
        target_module_to_kwargs = policy._run_policy(
            root_module, ignored_modules, root_kwargs
        )
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(
                root_module,
                mixed_precision._module_classes_to_ignore,
                ignored_modules,
                root_kwargs,
                target_module_to_kwargs,
            )
            overridden_module_classes = _override_module_mixed_precision(
                root_module, mixed_precision._module_classes_to_ignore
            )
            _warn_on_overridden_mixed_precision(overridden_module_classes)
        try:
            args = get_args()
            if args.tensor_model_parallel_size > 1:
                _run_tensor_parallel_pg_override_policy(
                    root_module,
                    {ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding},
                    ignored_modules,
                    root_kwargs,
                    target_module_to_kwargs,
                )
        except AssertionError:
            warnings.warn(
                "Global args is not correctly initialized, skip TP wrapping...")

        _validate_frozen_params(
            root_module,
            set(target_module_to_kwargs.keys()),
            ignored_params,
            True,
        )
        wrap_fn = _construct_wrap_fn(
            root_module, target_module_to_kwargs, fsdp_fn)
        _post_order_apply(root_module, wrap_fn)
        return

    recursive_wrap_kwargs = {
        "module": root_module,
        "auto_wrap_policy": policy,
        "wrapper_cls": fsdp_fn,
        "ignored_modules": ignored_modules,
        "ignored_params": ignored_params,
        "only_wrap_children": True,
    }
    if mixed_precision is not None:
        # Wrap modules of the ignored types separately and register forward
        # hooks to cast to fp32 and back to the original dtype, respectively
        overridden_module_classes = _override_module_mixed_precision(
            root_module, mixed_precision._module_classes_to_ignore
        )
        policy = functools.partial(
            _or_policy,
            policies=[
                policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
        )
        recursive_wrap_kwargs["auto_wrap_policy"] = policy
        _warn_on_overridden_mixed_precision(overridden_module_classes)
    # type: ignore[arg-type]
    _recursive_wrap(**recursive_wrap_kwargs, **root_kwargs)


def _check_nested_wrapping(root_module: nn.Module):
    for module_name, module in root_module.named_modules():
        if _get_module_zero3_state(module) is not None:
            raise ValueError(
                "FSDP auto wrapping requires modules to not already have "
                f"FSDP applied but found {module_name} in\n{root_module}"
            )
