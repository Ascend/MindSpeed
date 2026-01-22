# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, Any

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from mindspeed.fsdp.parallel_engine_config import FSDPPlanConfig
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match
from mindspeed.fsdp.utils.dtype import get_dtype


logger = logging.getLogger(__name__)


def fully_shard_parallel_modules(model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig):
    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)
    mp_policy = get_mixprecision_policy(fsdp_plan)
    config = {
        'mesh': fsdp_mesh,
        'ignored_params': ignored_params,
        'mp_policy': mp_policy,
    }

    for module, plan in fsdp_modules.items():
        module_config = config.copy()
        module_config.update(plan)
        fully_shard(module, **module_config)
    fully_shard(model, **config)
    set_modules_to_prefetch(model, fsdp_modules, fsdp_plan)
    return model


def set_modules_to_prefetch(model: torch.nn.Module, fsdp_modules: list[torch.nn.Module], fsdp_plan: FSDPPlanConfig):
    """Configure forward and backward prefetching."""
    wrapped_modules_in_order: list[torch.nn.Module] = []
    for sub_module in model.modules():  # pre-order
        if any(sub_module is target_module for target_module in fsdp_modules):
            wrapped_modules_in_order.append(sub_module)

    if fsdp_plan.num_to_forward_prefetch > 0:
        for i, layer in enumerate(wrapped_modules_in_order):
            j_end = min(len(wrapped_modules_in_order), i + 1 + fsdp_plan.num_to_forward_prefetch)
            layers_to_prefetch = wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

    if fsdp_plan.num_to_backward_prefetch > 0:
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        for i, layer in enumerate(rev_wrapped_modules_in_order):
            j_end = min(len(rev_wrapped_modules_in_order), i + 1 + fsdp_plan.num_to_backward_prefetch)
            layers_to_prefetch = rev_wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def get_mixprecision_policy(fsdp_plan: FSDPPlanConfig):
    """Construct the MixedPrecisionPolicy object."""
    param_dtype = get_dtype(fsdp_plan.param_dtype) if fsdp_plan.param_dtype else None
    reduce_dtype = get_dtype(fsdp_plan.reduce_dtype) if fsdp_plan.reduce_dtype else None
    output_dtype = get_dtype(fsdp_plan.output_dtype) if fsdp_plan.output_dtype else None

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=fsdp_plan.cast_forward_inputs
    )


def get_fsdp_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig, ignored_modules: Set[str]) -> dict[Any, Any]:
    fsdp_modules = {}
    for name, module in model.named_modules():
        for pattern, plan in fsdp_plan.apply_modules.items():
            if module_name_match(pattern, name) and name not in ignored_modules:
                print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 to module <{name}>')
                if module not in fsdp_modules:
                    fsdp_modules[module] = {}
                fsdp_modules.get(module).update(plan)
    if len(fsdp_modules) == 0:
        raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.apply_modules.keys()}.')
    return fsdp_modules


def get_ignored_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig):
    ignored_modules = set()
    ignored_params = set()
    for name, module in model.named_modules():
        for pattern in fsdp_plan.ignored_modules:
            if module_name_match(pattern, name):
                print_rank(logger.debug, f'[FSDP2]: Ignored module to apply fsdp2 <{name}>')
                ignored_modules.add(name)
                ignored_params.update(list(module.parameters(recurse=True)))
    return ignored_modules, ignored_params
