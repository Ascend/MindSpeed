# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    Any,
    Dict,
    Iterable,
    Set,
    Type,
)

import torch.nn as nn
from torch.distributed.fsdp.wrap import (
    _post_order_apply,
    _construct_wrap_fn,
    always_wrap_policy,
    _Policy,
    _module_wrap_policy,
    ModuleWrapPolicy,
    CustomPolicy,
    _run_mixed_precision_override_policy,
    _or_policy,
    _recursive_wrap,
    _wrap_module_cls_individually,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]


def _run_tensor_parallel_pg_override_policy(
    root_module: nn.Module,
    module_classes: Iterable[Type[nn.Module]],
    ignored_modules: Set[nn.Module],
    root_kwargs: Dict[str, Any],
    target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]],
):
    module_classes_tuple = tuple(set(module_classes))
    for module in root_module.modules():
        if module in ignored_modules:
            continue
        elif isinstance(module, module_classes_tuple):
            # This policy overrides any existing policy
            if module not in target_module_to_kwargs:
                # Only inherit from the root kwargs if not already specified
                target_module_to_kwargs[module] = root_kwargs
            target_module_to_kwargs[module]["process_group"] = root_kwargs["tp_zero_process_group"]
    return target_module_to_kwargs
