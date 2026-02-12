# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple, Any

import torch
import torch.nn as nn
from mindspeed.fsdp.quantization.mxfp8_config import MXFP8LinearConfig
from mindspeed.fsdp.quantization.utils import _QUANTIZE_CONFIG_HANDLER


def _replace_with_custom_fn_if_matches_filter(
        model,
        config,
        replacement_fn,
        filter_fn,
        cur_fqn="",
        device=None,
        *args,
) -> None:
    if filter_fn(model, cur_fqn[:-1], config):
        name = cur_fqn[:-1]
        if device is not None:
            model.to(device=device)  # move to device before quantization
            model = replacement_fn(model, config)
            return model
        else:
            return replacement_fn(model, config)
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter(
                child,
                config,
                replacement_fn,
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


def quantize_(
        model: nn.Module,
        config: MXFP8LinearConfig,
        filter_fn: Optional[Callable[[torch.nn.Module, str], bool]],
        device: Optional[torch.types.Device] = None,
):
    filter_fn = filter_fn
    handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
    saved_inv_freq = model.model.rotary_emb.inv_freq.detach().clone()

    try:
        _replace_with_custom_fn_if_matches_filter(
            model,
            config,
            handler,
            filter_fn,
            device=device,
        )
    except Exception as e:
        raise RuntimeError("Failed to replace model") from e

    model.model.rotary_emb.inv_freq = saved_inv_freq
