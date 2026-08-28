# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""HiF8 NaN/Inf Step Recovery.

Framework-layer transactional wrapper around ``train_step`` that detects
NaN/Inf in loss / gradient / optimizer overflow, syncs the failure decision
across all ranks, and retries the same step with the same batch after calling
the TransformerEngineNPU ``reset_fp8_amax_history`` interface.

The framework layer does NOT maintain or read any HiF8 warmup / CTS / DTS
state.  The only TransformerEngineNPU touch-point is
``FP8GlobalStateManager.reset_fp8_amax_history()`` invoked through
:class:`~.recovery.HiF8ResetAdapter`.

Only :class:`RecoverableStepRunner` is part of the public package interface;
all detectors, recovery primitives and diagnostics are internal implementation
details and should be imported from their submodules directly when needed
(e.g. for tests or custom injection).
"""

from mindspeed.core.transformer_engine.step_recovery.controller import RecoverableStepRunner

__all__ = [
    "RecoverableStepRunner",
]
