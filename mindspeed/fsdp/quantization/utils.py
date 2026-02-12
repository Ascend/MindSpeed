# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 定义layer_filter等函数
import functools
import re
from typing import Optional, Callable, Dict, Type
import torch
import torch.nn as nn
import torch_npu

from mindspeed.fsdp.quantization.mxfp8_config import QuantBaseConfig
from mindspeed.fsdp.utils.str_match import module_name_match

_QUANTIZE_CONFIG_HANDLER: Dict[
    Type[QuantBaseConfig],
    Callable[[torch.nn.Module, QuantBaseConfig], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    @functools.wraps(config_type)
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func
        return func

    return decorator


class TensorWithTranspose:
    def __init__(
            self,
            fp8_dtype: torch.dtype,
            data: torch.Tensor,
            scale: torch.Tensor,
            data_t: torch.Tensor,
            scale_t: torch.Tensor,
            dtype: torch.dtype = torch.float32,
    ):
        self.fp8_dtype = fp8_dtype
        self.data = data
        self.scale = scale
        self.data_t = data_t
        self.scale_t = scale_t
        self.ori_dtype = dtype

    def get_by_trans(self, transpose=False):
        if transpose:
            return self.data_t, self.scale_t
        return self.data, self.scale


def module_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    def ignored_modules(fqn: str, config: QuantBaseConfig):
        for pattern in config.mxfp8_ignored_modules:
            if module_name_match(pattern, fqn):
                return True
        return False

    if not isinstance(mod, nn.Linear):
        return False

    ignored_modules_flag = ignored_modules(fqn, config)
    if ignored_modules_flag:
        return False

    for pattern in config.mxfp8_apply_modules:
        m = re.match(r"(.*?layers\.\d+)", fqn)
        if m is not None:
            prefix = m.group(1)
            if module_name_match(pattern, prefix):
                return True
            else:
                return False

    return False
