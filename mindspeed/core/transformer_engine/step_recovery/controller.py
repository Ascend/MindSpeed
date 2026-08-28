# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""The recoverable step runner.

Orchestrates a single logical training step as a pre-commit transaction:

    execute attempt (with optimizer methods patched)
    -> pre-commit guard detects NaN/Inf before optimizer commit
    -> success: commit (clear replay cache) and return
    -> failure: cleanup -> HiF8 reset -> restore RNG -> replay batch -> retry
    -> retry still fails: dump and raise

The framework layer contains NO warmup / CTS / DTS logic.  The only
TransformerEngineNPU interaction is the
``FP8GlobalStateManager.reset_fp8_amax_history()`` call issued by the
:class:`~mindspeed.core.transformer_engine.step_recovery.recovery.HiF8ResetAdapter`.
"""

from logging import getLogger
from typing import Any, Callable, Optional

from mindspeed.core.transformer_engine.step_recovery.diagnostics import StepRecoveryDiagnostics
from mindspeed.core.transformer_engine.step_recovery.exceptions import RetryableStepError
from mindspeed.core.transformer_engine.step_recovery.pre_commit_guard import (
    StepHealth,
    make_guarded_forward_backward_func,
    patch_optimizer_pre_commit_methods,
)
from mindspeed.core.transformer_engine.step_recovery.recovery import (
    DistributedFailureSync,
    StepRecovery,
)

LOG = getLogger(__name__)


class RecoverableStepRunner:
    """Run a logical training step with pre-commit retry-on-NaN/Inf.

    On NaN/Inf detected before optimizer commit, clean up transient state,
    reset HiF8 amax history, restore RNG, replay the same batch and retry
    exactly once.  All optional collaborators can be injected for testing.
    """

    def __init__(
        self,
        failure_sync: Optional[DistributedFailureSync] = None,
        recovery: Optional[StepRecovery] = None,
        diagnostics: Optional[StepRecoveryDiagnostics] = None,
    ):
        self.failure_sync = failure_sync or DistributedFailureSync()
        self.recovery = recovery or StepRecovery()
        self.diagnostics = diagnostics or StepRecoveryDiagnostics()

    def run_train_step(
        self,
        *,
        original_train_step: Callable,
        logical_step: int,
        data_iterator: Any,
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
    ) -> Any:
        """Run original Megatron train_step with pre-commit recovery.

        This method does not copy train_step and does not rewrite optimizer.step.
        It temporarily patches optimizer methods so that NaN/Inf is caught
        *before* the optimizer commits parameter updates.
        """

        iteration = logical_step + 1
        attempt_iterator = self.recovery.prepare(data_iterator)

        for attempt in range(2):
            step_health = StepHealth()

            if attempt == 0:
                current_iterator = attempt_iterator
            else:
                current_iterator = self.recovery.get_retry_iterator()

            guarded_forward_backward_func = make_guarded_forward_backward_func(
                forward_backward_func,
                step_health,
            )

            try:
                with patch_optimizer_pre_commit_methods(
                    optimizer=optimizer,
                    failure_sync=self.failure_sync,
                    step_health=step_health,
                ):
                    result = original_train_step(
                        forward_step_func,
                        current_iterator,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        config,
                        guarded_forward_backward_func,
                        iteration=logical_step,
                    )

                self.recovery.commit()

                if attempt > 0:
                    self.diagnostics.log_retry_success(iteration, attempt)

                return result

            except RetryableStepError as exc:
                # If train_step raises before returning, the original post-step
                # path may not have called post_attempt().
                # Call it here so the retry iterator can replay the same batch.
                try:
                    self.recovery.post_attempt()
                except Exception:
                    LOG.debug("[StepRecovery] failed to save microbatches after retryable error", exc_info=True)

                self.diagnostics.log_retry_error(
                    iteration=iteration,
                    attempt=attempt,
                    error=exc,
                    global_retry=True,
                )

                if attempt == 1:
                    self.diagnostics.dump_abort_error(
                        iteration=iteration,
                        attempt=attempt,
                        error=exc,
                        context=None,
                    )
                    raise RuntimeError(
                        f"Iteration {iteration} failed after pre-commit retry "
                        f"(last reason: {exc.reason.name}, message={str(exc)})"
                    ) from exc

                self.recovery.recover(model=model, optimizer=optimizer)

        raise RuntimeError(f"Iteration {iteration} failed unexpectedly.")
