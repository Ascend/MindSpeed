# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

"""
This file includes public APIs for FSDP such as the classes used for the
constructor arguments.
"""


__all__ = [
    "BackwardPrefetch",
    "MixedPrecision",
]

from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type

import torch
from torch.nn.modules.batchnorm import _BatchNorm


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
