# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, Any

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh

from mindspeed.lite.mindspeed_lite_config import FSDPPlanConfig
from mindspeed.lite.utils.log import print_rank
from mindspeed.lite.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def fully_shard_parallel_modules(model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig):
    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)

    config = {'mesh': fsdp_mesh, 'ignored_params': ignored_params}
    for module, plan in fsdp_modules.items():
        module_config = config.copy()
        module_config.update(plan)
        fully_shard(module, **module_config)
    fully_shard(model, **config)
    return model


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
                ignored_params.update(list(module.parameters(recurse=False)))
    return ignored_modules, ignored_params
