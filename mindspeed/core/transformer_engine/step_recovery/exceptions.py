# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Exceptions for HiF8 pre-commit step recovery."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class RetryReason(IntEnum):
    NONE = 0
    OPTIMIZER_OVERFLOW = 1
    LOSS_INF = 2
    LOSS_NAN = 3
    GRAD_INF = 4
    GRAD_NAN = 5


@dataclass
class RetryableStepError(RuntimeError):
    """Raised before optimizer commit when the current logical step must retry."""

    reason: RetryReason = RetryReason.NONE
    message: str = ""
    loss_value: Optional[float] = None
    grad_norm_value: Optional[float] = None

    def __str__(self):
        return self.message
