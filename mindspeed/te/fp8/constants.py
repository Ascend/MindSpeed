from enum import Enum

import torch
import torch_npu

# FP8 Dtype need new torch npu
if not hasattr(torch, 'float8_e4m3fn') or not hasattr(torch, 'float8_e5m2'):
    torch.float8_e4m3fn = torch.bfloat16
    torch.float8_e5m2 = torch.bfloat16


class FP8Format:
    def __init__(self, range_max: float, ebits: int, mbits: int, dtype: torch.dtype):
        self.max = range_max
        self.ebits = ebits
        self.mbits = mbits
        self.dtype = dtype


class Format(Enum):
    E4M3 = FP8Format(448, 4, 3, torch.float8_e4m3fn)
    E5M2 = FP8Format(57344, 5, 2, torch.float8_e5m2)
    HiF8 = FP8Format(57344, 5, 2, None)


def amax_compute_max(amax, amax_history):
    amax.copy_(torch.amax(amax_history, dim=0))


AMAX_COMPUTE_MAP = {
    'max': amax_compute_max
}
