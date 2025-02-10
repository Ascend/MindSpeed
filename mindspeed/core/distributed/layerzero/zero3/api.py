"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""

from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type

import torch
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    "BackwardPrefetch",
    "MixedPrecision",
]


class BackwardPrefetch(Enum):
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


class BackwardReduceScatter(Enum):
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


@dataclass
class MixedPrecision:
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    buffer_dtype: Optional[torch.dtype] = None
    _module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = (_BatchNorm,)
