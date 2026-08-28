# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Diagnostics for the step recovery feature.

Emits structured log lines for each retry, each HiF8 reset, and the final
abort when retries are exhausted.  Never reads or prints TE-internal warmup
state.
"""

import json
from logging import getLogger
from typing import Any, Optional

LOG = getLogger(__name__)


def _rank() -> int:
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        LOG.debug("[StepRecovery] failed to get rank", exc_info=True)
    return 0


def _is_rank_zero() -> bool:
    return _rank() == 0


class StepRecoveryDiagnostics:
    """Structured logging for step recovery events.

    ``iteration`` is the 1-based user-facing iteration number (i.e. the
    internal 0-based ``logical_step`` plus one), matching the convention used
    by Megatron's ``training_log``.
    """

    def log_retry_error(
        self,
        *,
        iteration: int,
        attempt: int,
        error,
        global_retry: bool = True,
    ) -> None:
        """Log a retry event using a RetryableStepError."""
        if _is_rank_zero():
            LOG.warning(
                "[StepRecovery] retry iteration=%s attempt=%s reason=%s "
                "loss=%s grad_norm=%s global_retry=%s message=%s",
                iteration,
                attempt,
                error.reason.name,
                error.loss_value,
                error.grad_norm_value,
                global_retry,
                str(error),
            )

    def log_retry_success(self, iteration: int, attempt: int) -> None:
        if _is_rank_zero():
            LOG.info(
                "[StepRecovery] iteration=%s retry succeeded on attempt=%s",
                iteration,
                attempt,
            )

    def dump_abort_error(
        self,
        *,
        iteration: int,
        attempt: int,
        error,
        context: Optional[Any] = None,
    ) -> None:
        """Dump abort diagnostics using a RetryableStepError."""
        entry = {
            "event": "abort",
            "iteration": int(iteration),
            "attempt": int(attempt),
            "reason": error.reason.name,
            "rank": _rank(),
            "loss": error.loss_value,
            "grad_norm": error.grad_norm_value,
            "context": repr(context) if context is not None else None,
        }
        # Dump on every rank so the abort is diagnosable from any log.
        LOG.error(
            "[StepRecovery] abort iteration=%s attempt=%s reason=%s loss=%s grad_norm=%s message=%s context=%s",
            iteration,
            attempt,
            error.reason.name,
            error.loss_value,
            error.grad_norm_value,
            str(error),
            context,
        )
        try:
            LOG.error("[StepRecovery] event_log=%s", json.dumps(entry))
        except Exception:
            LOG.debug("[StepRecovery] failed to dump event_log", exc_info=True)
