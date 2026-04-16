# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import re
import torch
import torch.nn as nn

from mindspeed.fsdp.parallel_engine_config import QuantizeConfig
from mindspeed.fsdp.utils.str_match import module_name_match


def _replace_with_custom_fn_if_matches_filter(
        model,
        config,
        convert_fn,
        filter_fn,
        cur_fqn="",
        device=None,
        *args,
) -> None:
    if filter_fn(model, cur_fqn[:-1], config):
        name = cur_fqn[:-1]
        if device is not None:
            model.to(device=device)  # move to device before quantization
            model = convert_fn(model, config, name)
            return model
        else:
            return convert_fn(model, config, name)
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter(
                child,
                config,
                convert_fn,
                filter_fn,
                f"{cur_fqn}{name}.",
                device=device,
                *args,
            )
            if new_child is not child and new_child is not None:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)
        return model


def convert_model(
        model: nn.Module,
        config: QuantizeConfig,
        convert_fn: Callable[[torch.nn.Module, QuantizeConfig], torch.nn.Module],
        filter_fn: Optional[Callable[[torch.nn.Module, str], bool]],
        device: Optional[torch.types.Device] = None,
):
    saved_inv_freq = model.model.rotary_emb.inv_freq.detach().clone()

    try:
        _replace_with_custom_fn_if_matches_filter(
            model,
            config,
            convert_fn,
            filter_fn,
            device=device,
        )
    except Exception as e:
        raise RuntimeError("Failed to replace model") from e

    model.model.rotary_emb.inv_freq = saved_inv_freq


def module_filter_fn(mod: nn.Module, fqn: str, config: QuantizeConfig) -> bool:
    def ignored_modules(fqn: str, config: QuantizeConfig):
        for pattern in config.ignored_modules:
            if module_name_match(pattern, fqn):
                return True
        return False

    if not isinstance(mod, nn.Linear):
        return False

    ignored_modules_flag = ignored_modules(fqn, config)
    if ignored_modules_flag:
        return False

    for pattern in config.apply_modules:
        m = re.match(r"(.*?layers\.\d+)", fqn)
        if m is not None:
            prefix = m.group(1)
            if module_name_match(pattern, prefix):
                return True
            else:
                return False

    return False


def moe_filter_fn(mod: nn.Module, fqn: str, config) -> bool:
    if "experts" not in fqn.lower():
        return False

    m = re.match(r"(.*?layers\.\d+)", fqn)
    if m is None:
        return False
    for pattern in config.apply_modules:
        m = re.match(r"(.*?layers\.\d+)", fqn)
        if m is not None:
            prefix = m.group(1)
            if module_name_match(pattern, prefix):
                return True
            else:
                return False

    return False
