# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import types
from functools import partial
from typing import Callable

import torch

from mindspeed.lite.distributed.expert_parallel.dispatcher import ep_forward
from mindspeed.lite.distributed.expert_parallel.dispatcher_mc2 import ep_forward as mc2_ep_forward
from mindspeed.lite.mindspeed_lite_config import EPPlanConfig
from mindspeed.lite.utils.log import print_rank
from mindspeed.lite.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def expert_parallelize_modules(modules: torch.nn.Module, ep_group, plan: EPPlanConfig):
    ep_modules = get_ep_modules(modules, plan)

    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    for module in ep_modules:
        # calculate local experts id
        module.num_global_experts = len(module) if not hasattr(module, 'num_experts') else module.num_experts
        if module.num_global_experts % ep_size != 0:
            raise AssertionError(
                f'Number of experts({module.num_global_experts}) is not divisible by ep size({ep_size}).')
        module.num_local_experts = module.num_global_experts // ep_size
        local_expert_indices_offset = ep_rank * module.num_local_experts
        module.local_expert_indices = [local_expert_indices_offset + i for i in range(module.num_local_experts)]
        if module.num_local_experts > 1:
            module.expert_ids_per_ep_rank = torch.tensor(
                [i % module.num_local_experts for i in range(module.num_global_experts)], dtype=torch.int32,
                device=torch.accelerator.current_device_index())

        # replace global experts with local experts
        local_experts = []
        for i in range(module.local_expert_indices[0], module.local_expert_indices[-1] + 1):
            local_experts.append(module[i])
        while len(module) > 0:
            module.pop(0)
        module.extend(local_experts)

        # replace forward with ep forward
        forward_fn = get_dispatcher_fn(plan.dispatcher, ep_group)
        module.forward = types.MethodType(forward_fn, module)

        # apply ep parameter grad division, if efsdp is enabled, the hook will be overridden
        apply_grad_division_hook(module, ep_size)

    return modules


def get_ep_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    ep_modules = []
    for plan_name in plan.apply_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Parallel]: Apply efsdp to module <{name}>')
                ep_modules.append(module)
    if len(ep_modules) == 0:
        raise RuntimeError(f'[Expert Parallel] No module named {plan} or not be ModuleList')
    return ep_modules


def prepare_total_weights(local_experts, module):
    module.gate_weights = []
    module.up_weights = []
    module.down_weights = []
    for mlp in local_experts:
        module.gate_weights.append(mlp.gate_proj.weight)
        module.up_weights.append(mlp.up_proj.weight)
        module.down_weights.append(mlp.down_proj.weight)


def get_dispatcher_fn(dispatcher, ep_group):
    forward_fn = None
    if isinstance(dispatcher, Callable):
        forward_fn = partial(dispatcher, ep_group)
    elif isinstance(dispatcher, str):
        if dispatcher == 'eager':
            forward_fn = partial(ep_forward, ep_group, False)
        elif dispatcher == 'fused':
            forward_fn = partial(ep_forward, ep_group, True)
        elif dispatcher == 'mc2':
            forward_fn = partial(mc2_ep_forward, ep_group)

    if forward_fn is None:
        raise RuntimeError(f'Unsupported dispatcher {dispatcher}.')

    return forward_fn


def get_grad_division_hook(param, ep_size):
    def hook(*unused):
        return param.grad.mul_(1 / ep_size)

    return hook


def apply_grad_division_hook(module, ep_size):
    for param in module.parameters():
        if param.requires_grad:
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(get_grad_division_hook(param, ep_size))