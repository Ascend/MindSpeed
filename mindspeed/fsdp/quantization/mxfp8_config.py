# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass
from enum import Enum
from typing import Optional, NamedTuple
from dataclasses import field

import torch
import torch_npu

from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class FP8Format:
    def __init__(self, range_max: float, ebits: int, mbits: int, dtype: Optional[torch.dtype]):
        self.max = range_max
        self.ebits = ebits
        self.mbits = mbits
        self.dtype = dtype

    @property
    def quant_type(self):
        if self.dtype is None:
            return torch_npu.hifloat8
        return self.dtype


class FormatEnum(Enum):
    E4M3 = FP8Format(448, 4, 3, torch.float8_e4m3fn)
    E5M2 = FP8Format(57344, 5, 2, torch.float8_e5m2)
    HIF8 = FP8Format(57344, 5, 2, None)


class _FormatConfig(NamedTuple):
    inputs: FormatEnum = FormatEnum.E4M3
    weight: FormatEnum = FormatEnum.E4M3
    grads: FormatEnum = FormatEnum.E4M3


class Format(Enum):
    E4M3 = _FormatConfig()
    HYBRID = _FormatConfig(grads=FormatEnum.E5M2)
    HIF8 = _FormatConfig(
        inputs=FormatEnum.HIF8,
        weight=FormatEnum.HIF8,
        grads=FormatEnum.HIF8
    )

    @classmethod
    def from_config_fp8(cls, key: str):
        return getattr(cls, key.upper(), None)


@dataclass
class QuantBaseConfig:
    quant_format: str = "E4M3"
    block_size: int = 32

    @classmethod
    def get_key_dtype(cls, key):
        key_format = Format.from_config_fp8(cls.quant_format)
        config = key_format.value
        if key == 'inputs':
            return config.inputs.value.quant_type
        elif key == 'weight':
            return config.weight.value.quant_type
        else:
            return config.grads.value.quant_type


@dataclass
class MXFP8LinearConfig(QuantBaseConfig):
    mxfp8_ignored_modules: list[str] = field(default_factory=list)
    mxfp8_apply_modules: list[str] = field(default_factory=list)
    converter: str = "quantize.linear.mx"


def get_mxfp8linear_config(quant_config: QuantizeConfig):
    return MXFP8LinearConfig(
        quant_format=quant_config.quant_format,
        block_size=quant_config.block_size,
        mxfp8_ignored_modules=quant_config.quant_ignored_modules,
        mxfp8_apply_modules=quant_config.quant_apply_modules
    )
