import typing
from enum import Enum
from typing import Optional, NamedTuple

import torch

import torch_npu

# FP8 Dtype need new torch npu
if not hasattr(torch, 'float8_e4m3fn') or not hasattr(torch, 'float8_e5m2'):
    torch.float8_e4m3fn = torch.bfloat16
    torch.float8_e5m2 = torch.bfloat16


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


class Fp8Recipe(str, Enum):
    delayed = 'delayed'
    tensorwise = 'tensorwise'
    mxfp8 = 'mxfp8'
    blockwise = 'blockwise'
    groupwise = 'groupwise'


def amax_compute_max(amax, amax_history, last_history_index):
    amax.copy_(torch.amax(amax_history, dim=0))


def amax_compute_most_recent(amax: torch.Tensor, amax_history, last_history_index):
    amax.copy_(amax_history[last_history_index])


AMAX_COMPUTE_MAP = {
    'max': amax_compute_max,
    "most_recent": amax_compute_most_recent
}
