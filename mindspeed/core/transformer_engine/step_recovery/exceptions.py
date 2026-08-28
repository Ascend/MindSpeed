# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
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
