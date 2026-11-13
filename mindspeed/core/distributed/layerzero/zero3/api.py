# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
