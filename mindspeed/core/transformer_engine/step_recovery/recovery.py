# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Recovery primitives for HiF8 step recovery.

This module groups all the collaborators used by
:class:`~mindspeed.core.transformer_engine.step_recovery.controller.RecoverableStepRunner`
to recover from a failed attempt:

* :class:`DistributedFailureSync`     - all-rank failure sync via all_reduce(MAX)
* :class:`FailedAttemptCleanup`       - clear grads / overflow flags / async comms
* :class:`HiF8ResetAdapter`           - ask TE to reset HiF8 amax history
* :class:`RNGStateManager`            - capture / restore RNG state
* :class:`ExternalReplayAdapter` /
  :class:`NativeRerunReplayAdapter` /
  :func:`create_replay_adapter`       - replay the same batch on retry

The framework layer contains NO warmup / CTS / DTS logic.  The only
TransformerEngineNPU interaction is the
``FP8GlobalStateManager.reset_fp8_amax_history()`` call issued by
:class:`HiF8ResetAdapter`.
"""

from logging import getLogger
from typing import Any, Dict, Iterator, List, Optional, Union

import torch

from mindspeed.core.transformer_engine.step_recovery.diagnostics import _is_rank_zero
from mindspeed.core.transformer_engine.step_recovery.exceptions import RetryReason

LOG = getLogger(__name__)


# ===========================================================================
# Distributed failure synchronization
# ===========================================================================


class DistributedFailureSync:
    """Synchronize failure flag and failure reason across all ranks.

    Uses all_reduce(MAX) on a small int32 payload:
      payload[0] = failure flag
      payload[1] = RetryReason value

    This makes the retry decision and diagnostic reason identical on every rank.
    """

    def __init__(self, group=None):
        self.group = group

    def sync_failure(
        self,
        local_failed: bool,
        local_reason: RetryReason = RetryReason.NONE,
    ):
        """Return (global_failed, global_reason)."""

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            if local_failed:
                return True, local_reason
            return False, RetryReason.NONE

        device = self._pick_device()

        payload = torch.tensor(
            [
                1 if local_failed else 0,
                int(local_reason) if local_failed else int(RetryReason.NONE),
            ],
            device=device,
            dtype=torch.int32,
        )

        torch.distributed.all_reduce(
            payload,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )

        global_failed = bool(payload[0].item())
        global_reason = RetryReason(int(payload[1].item()))

        return global_failed, global_reason

    def any_rank_failed(self, local_failed: bool) -> bool:
        """Backward-compatible bool-only sync.

        Keep this method for any old internal tests, but new pre-commit code
        should use sync_failure() so it can also report global_reason.
        """

        global_failed, _ = self.sync_failure(
            local_failed,
            RetryReason.OPTIMIZER_OVERFLOW if local_failed else RetryReason.NONE,
        )
        return global_failed

    @staticmethod
    def _pick_device():
        """Pick a device for the all-reduce flag tensor.

        On NPU torch_npu.contrib.transfer_to_npu aliases torch.cuda to
        the NPU device, so torch.cuda.current_device() is the right call.
        """
        try:
            if torch.cuda.is_available():
                return torch.cuda.current_device()
        except Exception:
            LOG.debug("[StepRecovery] failed to pick CUDA device", exc_info=True)
        return torch.device("cpu")


# ===========================================================================
# Transient-state cleanup
# ===========================================================================


class FailedAttemptCleanup:
    """Clean up transient state after a failed attempt.

    Clears gradients (main_grad / grad buffer / fused accumulation buffer),
    optimizer attempt-local overflow flags, and any pending async
    communication.  Parameter updates, optimizer momentum / variance,
    scheduler state, global step and consumed-sample counters are **never**
    touched here: by design they are only committed after a successful
    attempt.
    """

    def cleanup(self, *, model: Optional[Any] = None, optimizer: Optional[Any] = None) -> None:
        self._clear_gradients(model, optimizer)
        self._clear_optimizer_overflow_flags(optimizer)
        self._synchronize_async()

    @staticmethod
    def _clear_gradients(model: Optional[Any], optimizer: Optional[Any]) -> None:
        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                try:
                    optimizer.zero_grad()
                except Exception:
                    LOG.debug("[StepRecovery] optimizer.zero_grad() fallback failed", exc_info=True)
            except Exception:
                LOG.debug("[StepRecovery] optimizer.zero_grad(set_to_none=True) failed", exc_info=True)

        if model is None:
            return

        model_list = model if isinstance(model, (list, tuple)) else [model]
        for chunk in model_list:
            zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
            if callable(zero_grad_buffer):
                try:
                    zero_grad_buffer()
                except Exception:
                    LOG.debug("[StepRecovery] chunk.zero_grad_buffer() failed", exc_info=True)
            try:
                chunk.zero_grad(set_to_none=True)
            except TypeError:
                try:
                    chunk.zero_grad()
                except Exception:
                    LOG.debug("[StepRecovery] chunk.zero_grad() fallback failed", exc_info=True)
            except Exception:
                LOG.debug("[StepRecovery] chunk.zero_grad(set_to_none=True) failed", exc_info=True)

    @staticmethod
    def _clear_optimizer_overflow_flags(optimizer: Optional[Any]) -> None:
        if optimizer is None:
            return
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is not None and torch.is_tensor(found_inf):
            try:
                found_inf.fill_(0.0)
            except Exception:
                LOG.debug("[StepRecovery] failed to clear optimizer found_inf", exc_info=True)
        chained = getattr(optimizer, "chained_optimizers", None)
        if chained:
            for sub in chained:
                sub_found = getattr(sub, "found_inf", None)
                if sub_found is not None and torch.is_tensor(sub_found):
                    try:
                        sub_found.fill_(0.0)
                    except Exception:
                        LOG.debug("[StepRecovery] failed to clear chained optimizer found_inf", exc_info=True)

    @staticmethod
    def _synchronize_async() -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            LOG.debug("[StepRecovery] torch.cuda.synchronize() failed", exc_info=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                LOG.debug("[StepRecovery] torch.distributed.barrier() failed", exc_info=True)


# ===========================================================================
# HiF8 amax history reset
# ===========================================================================


class HiF8ResetAdapter:
    """Adapter that requests an HiF8 amax-history reset from TE.

    Best-effort: if TransformerEngineNPU / the reset API is unavailable
    (e.g. running without HiF8 / FP8), a warning is logged and the call is
    skipped so the rest of the recovery flow still proceeds.
    """

    def reset_after_failed_step(self) -> None:
        """Reset HiF8 amax history on every rank after a confirmed failure."""
        try:
            from transformer_engine.pytorch.quantization import (
                FP8GlobalStateManager,
            )
        except ImportError:
            LOG.warning(
                "[StepRecovery] transformer_engine FP8GlobalStateManager not "
                "available; skipping HiF8 amax history reset."
            )
            return

        reset_fn = getattr(FP8GlobalStateManager, "reset_fp8_amax_history", None)

        if reset_fn is None:
            LOG.warning("[StepRecovery] no HiF8/FP8 amax history reset API found; skipping HiF8 amax history reset.")
            return

        reset_fn()
        if _is_rank_zero():
            LOG.info("[StepRecovery] HiF8 amax history reset requested")


# ===========================================================================
# RNG state capture / restore
# ===========================================================================


class RNGStateManager:
    """Capture and restore all relevant RNG state.

    Captures Python, NumPy, Torch CPU and device (NPU/CUDA) RNG, plus the
    Megatron tensor-parallel RNG tracker states so that dropout / random
    masks / MoE routing are reproduced as closely as possible on retry.
    """

    def capture(self) -> Dict[str, Any]:
        import random

        snapshot: Dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
        }

        try:
            import numpy as np

            snapshot["numpy"] = np.random.get_state()
        except Exception:
            LOG.debug("[StepRecovery] failed to capture numpy RNG state", exc_info=True)

        try:
            if torch.cuda.is_available():
                snapshot["torch_device"] = torch.cuda.get_rng_state()
        except Exception:
            LOG.debug("[StepRecovery] failed to capture CUDA RNG state", exc_info=True)

        snapshot["tp_rng_tracker"] = self._capture_tp_rng_tracker()
        return snapshot

    def restore(self, snapshot: Dict[str, Any]) -> None:
        import random

        if "python" in snapshot:
            random.setstate(snapshot["python"])
        if "torch_cpu" in snapshot:
            torch.set_rng_state(snapshot["torch_cpu"])
        if "numpy" in snapshot:
            try:
                import numpy as np

                np.random.set_state(snapshot["numpy"])
            except Exception:
                LOG.debug("[StepRecovery] failed to restore numpy RNG state", exc_info=True)
        if "torch_device" in snapshot:
            try:
                torch.cuda.set_rng_state(snapshot["torch_device"])
            except Exception:
                LOG.debug("[StepRecovery] failed to restore CUDA RNG state", exc_info=True)
        if "tp_rng_tracker" in snapshot:
            self._restore_tp_rng_tracker(snapshot["tp_rng_tracker"])

    @staticmethod
    def _capture_tp_rng_tracker() -> Optional[Any]:
        try:
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

            tracker = get_cuda_rng_tracker()
            if tracker.is_initialized():
                return tracker.get_states()
        except Exception:
            LOG.debug("[StepRecovery] failed to capture TP RNG tracker state", exc_info=True)
        return None

    @staticmethod
    def _restore_tp_rng_tracker(states: Optional[Any]) -> None:
        if not states:
            return
        try:
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

            tracker = get_cuda_rng_tracker()
            tracker.set_states(states)
        except Exception:
            LOG.debug("[StepRecovery] failed to restore TP RNG tracker state", exc_info=True)


# ===========================================================================
# Batch replay
# ===========================================================================


def _maybe_clone(item: Any) -> Any:
    """Best-effort deep-ish copy of a microbatch.

    ``get_batch_on_this_tp_rank`` typically returns a dict of tensors that may
    be written to (e.g. shifted in place).  Clone the tensors we can identify
    so the replayed batch is identical to the original.
    """
    if item is None:
        return None
    if isinstance(item, dict):
        return {k: _maybe_clone(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        cloned = [_maybe_clone(v) for v in item]
        return type(item)(cloned) if isinstance(item, tuple) else cloned
    if torch.is_tensor(item):
        try:
            return item.clone()
        except Exception:
            return item
    return item


class ReplayableIterator:
    """A single replayable iterator wrapping a source iterator.

    Caches every consumed item and replays on rollback.  Works because
    ``_sanitize_data_iterators`` returns an empty list when rerun mode is
    ``DISABLED``, so no isinstance assert fires.
    """

    def __init__(self, source: Iterator):
        self.source = source
        self._cache: List[Any] = []
        self._replay_index = 0
        self._replay_mode = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._replay_mode:
            if self._replay_index >= len(self._cache):
                raise RuntimeError(
                    "ReplayableIterator exhausted its cache: the retry is "
                    "consuming more microbatches than the original attempt. "
                    "This indicates a divergence between attempts."
                )
            item = self._cache[self._replay_index]
            self._replay_index += 1
            return _maybe_clone(item)

        item = next(self.source)
        self._cache.append(_maybe_clone(item))
        return item

    def commit(self):
        """Discard the cache after a successful attempt."""
        self._cache = []
        self._replay_mode = False
        self._replay_index = 0

    def rollback(self):
        """Prepare for a replay of the cached items."""
        self._replay_mode = True
        self._replay_index = 0


class ExternalReplayAdapter:
    """Manage replay for a single optimizer step using :class:`ReplayableIterator`.

    ``data_iterator`` may be a single iterator or a list of iterators.
    ``None`` entries are preserved as-is so that pipeline ranks that don't
    consume data retain their ``data_iterator is None`` semantics.
    """

    def __init__(self, data_iterator: Union[Iterator, List[Iterator]]):
        if isinstance(data_iterator, (list, tuple)):
            self._iterators: List[Any] = [None if it is None else ReplayableIterator(it) for it in data_iterator]
            self._was_list = True
        else:
            self._iterators = [None if data_iterator is None else ReplayableIterator(data_iterator)]
            self._was_list = False

    def get_original_iterator(self):
        """Return the iterator (or list) for the original attempt."""
        if self._was_list:
            return list(self._iterators)
        return self._iterators[0]

    def get_retry_iterator(self):
        """Return the iterator (or list) for the retry attempt."""
        for it in self._iterators:
            if it is not None:
                it.rollback()
        if self._was_list:
            return list(self._iterators)
        return self._iterators[0]

    def commit(self):
        for it in self._iterators:
            if it is not None:
                it.commit()

    def rollback(self):
        for it in self._iterators:
            if it is not None:
                it.rollback()


class NativeRerunReplayAdapter:
    """Replay using ``RerunDataIterator.saved_microbatches``.

    When Megatron's rerun state machine is active (mode != DISABLED), the data
    iterators are already :class:`RerunDataIterator` instances and
    ``_sanitize_data_iterators`` asserts this.  After attempt 0 we read
    ``saved_microbatches`` (populated by ``RerunDataIterator.__next__``) and
    clone them; for retry, we construct a *new* ``RerunDataIterator`` from the
    cloned list.
    """

    def __init__(self, data_iterator):
        from megatron.core.rerun_state_machine import RerunDataIterator

        self._RerunDataIterator = RerunDataIterator
        self._was_list = isinstance(data_iterator, (list, tuple))

        if self._was_list:
            self._original_iters: List[Any] = list(data_iterator)
        else:
            self._original_iters = [data_iterator]

        self._cached_batches: Optional[List[List[Any]]] = None

    def get_original_iterator(self):
        """Return the original iterator (or list) for the first attempt."""
        return list(self._original_iters) if self._was_list else self._original_iters[0]

    def get_retry_iterator(self):
        """Return a new RerunDataIterator (or list) for the retry attempt."""
        if self._cached_batches is None:
            raise RuntimeError(
                "NativeRerunReplayAdapter: get_retry_iterator() called "
                "before post_attempt() extracted saved_microbatches."
            )

        replay_iters = [self._RerunDataIterator(iter(mb_list)) for mb_list in self._cached_batches]
        return replay_iters if self._was_list else replay_iters[0]

    def post_attempt(self):
        """Called after attempt 0 completes to extract and clone microbatches."""
        self._cached_batches = []
        for it in self._original_iters:
            if it is None:
                self._cached_batches.append([])
                continue
            saved = getattr(it, "saved_microbatches", [])
            self._cached_batches.append([_maybe_clone(mb) for mb in saved])

    def commit(self):
        """Advance (clear) the original iterators after success."""
        for it in self._original_iters:
            if it is not None and hasattr(it, "advance"):
                it.advance()
        self._cached_batches = None

    def rollback(self):
        """No-op: cached_batches are already saved and immutable."""
        pass


def create_replay_adapter(data_iterator):
    """Return the appropriate replay adapter.

    Selection logic:
    - If ``data_iterator`` is a ``RerunDataIterator`` (or list thereof) AND
      rerun mode != DISABLED -> :class:`NativeRerunReplayAdapter`.
    - Otherwise -> :class:`ExternalReplayAdapter`.
    """
    try:
        from megatron.core.rerun_state_machine import (
            RerunDataIterator,
            get_rerun_state_machine,
            RerunMode,
        )
    except Exception:
        return ExternalReplayAdapter(data_iterator)

    if isinstance(data_iterator, RerunDataIterator):
        has_rerun = True
    elif isinstance(data_iterator, (list, tuple)):
        has_rerun = any(isinstance(d, RerunDataIterator) for d in data_iterator if d is not None)
    else:
        has_rerun = False

    if has_rerun:
        try:
            rsm = get_rerun_state_machine()
            if rsm.get_mode() != RerunMode.DISABLED:
                return NativeRerunReplayAdapter(data_iterator)
        except Exception:
            LOG.debug(
                "[StepRecovery] failed to query rerun state machine; falling back to external replay adapter",
                exc_info=True,
            )

    return ExternalReplayAdapter(data_iterator)


# ===========================================================================
# StepRecovery - single high-level facade used by the controller
# ===========================================================================


class StepRecovery:
    """Coordinate all recovery primitives for one logical step.

    Wraps cleanup, HiF8 amax reset, RNG restore and batch replay behind a
    small imperative API so the controller does not need to know about the
    individual collaborators:

    ::

        recovery = StepRecovery()
        it = recovery.prepare(data_iterator)   # capture RNG + build replay
        result = step_fn(data_iterator=it)
        recovery.post_attempt()                 # save microbatches for replay
        if success:
            recovery.commit()                   # discard replay cache
        else:
            recovery.recover(model, optimizer)  # cleanup + reset + RNG + rollback
            it = recovery.get_retry_iterator()
            result = step_fn(data_iterator=it)
    """

    def __init__(
        self,
        cleanup: Optional[FailedAttemptCleanup] = None,
        hif8_adapter: Optional[HiF8ResetAdapter] = None,
        rng_manager: Optional[RNGStateManager] = None,
    ):
        self._cleanup = cleanup or FailedAttemptCleanup()
        self._hif8 = hif8_adapter or HiF8ResetAdapter()
        self._rng = rng_manager or RNGStateManager()
        self._replay = None
        self._rng_snapshot: Optional[Dict[str, Any]] = None

    def prepare(self, data_iterator: Any) -> Any:
        """Capture RNG state and build the replay adapter.

        Returns the iterator (or list) to feed into the original attempt.
        """
        self._replay = create_replay_adapter(data_iterator)
        self._rng_snapshot = self._rng.capture()
        return self._replay.get_original_iterator()

    def post_attempt(self) -> None:
        """Save microbatches after the original attempt (native rerun mode)."""
        if hasattr(self._replay, "post_attempt"):
            self._replay.post_attempt()

    def recover(self, *, model: Any = None, optimizer: Any = None) -> None:
        """Run the full recovery sequence before a retry.

        Order: cleanup transient state -> reset HiF8 amax history -> restore
        RNG -> rollback the replay adapter.
        """
        self._cleanup.cleanup(model=model, optimizer=optimizer)
        self._hif8.reset_after_failed_step()
        if self._rng_snapshot is not None:
            self._rng.restore(self._rng_snapshot)
        self._replay.rollback()

    def get_retry_iterator(self) -> Any:
        """Return the iterator to feed into the retry attempt."""
        return self._replay.get_retry_iterator()

    def commit(self) -> None:
        """Discard the replay cache after a successful attempt."""
        self._replay.commit()
