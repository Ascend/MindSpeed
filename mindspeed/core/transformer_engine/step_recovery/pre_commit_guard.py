# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Pre-commit recovery guard for HiF8 step recovery.

This module temporarily patches optimizer methods during one ``train_step``
call so that NaN/Inf in loss / grad_norm / found_inf is caught *before* the
optimizer commits parameter updates.

It does NOT copy ``train_step`` and does NOT rewrite ``optimizer.step()``.
Only top-level optimizer methods are patched.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any, List, Optional, Tuple

import math
import torch

from mindspeed.core.transformer_engine.step_recovery.exceptions import (
    RetryableStepError,
    RetryReason,
)

LOG = getLogger(__name__)


@dataclass
class StepHealth:
    """Record the health status of the current step attempt."""

    invalid: bool = False
    reason: RetryReason = RetryReason.NONE
    loss_value: Optional[float] = None
    grad_norm_value: Optional[float] = None

    def mark_loss_invalid(self, reason: RetryReason, value: Optional[float]):
        self.invalid = True
        self.reason = reason
        self.loss_value = value

    def mark_grad_norm_invalid(self, reason: RetryReason, value: Optional[float]):
        self.invalid = True
        self.reason = reason
        self.grad_norm_value = value

    def mark_overflow(self):
        self.invalid = True
        self.reason = RetryReason.OPTIMIZER_OVERFLOW


def _is_non_finite(value: Any) -> bool:
    if value is None:
        return False

    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return False
            return not torch.isfinite(value).all().item()

        return not torch.isfinite(torch.as_tensor(float(value))).item()
    except Exception:
        return False


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())

        return float(value)
    except Exception:
        return None


def _classify(value: Any, nan_reason: RetryReason, inf_reason: RetryReason):
    """Classify a non-finite value as NaN or Inf."""

    scalar = _to_float(value)

    try:
        if torch.is_tensor(value):
            tensor_value = value.detach()
        else:
            tensor_value = torch.as_tensor(value)

        if torch.isnan(tensor_value).any().item():
            return nan_reason, scalar

        if torch.isinf(tensor_value).any().item():
            return inf_reason, scalar
    except Exception:
        LOG.debug("[StepRecovery] failed to classify loss/grad_norm value", exc_info=True)

    if scalar is not None and math.isnan(scalar):
        return nan_reason, scalar

    return inf_reason, scalar


def _extract_loss_items(losses_reduced, prefix: str = "loss") -> List[Tuple[str, Any]]:
    """Extract scalar-like loss items from Megatron forward_backward return.

    Supports:
      - dict: {"lm loss": tensor}
      - tuple/list: (loss, num_tokens, loss_reduced)
      - nested tuple/list/dict
    """

    items: List[Tuple[str, Any]] = []

    if losses_reduced is None:
        return items

    if isinstance(losses_reduced, dict):
        for key, value in losses_reduced.items():
            items.extend(_extract_loss_items(value, f"{prefix}.{key}"))
        return items

    if isinstance(losses_reduced, (list, tuple)):
        for idx, value in enumerate(losses_reduced):
            items.extend(_extract_loss_items(value, f"{prefix}.{idx}"))
        return items

    if torch.is_tensor(losses_reduced):
        items.append((prefix, losses_reduced))
        return items

    if isinstance(losses_reduced, (float, int)):
        items.append((prefix, losses_reduced))
        return items

    return items


def make_guarded_forward_backward_func(forward_backward_func, step_health: StepHealth):
    """Record loss NaN/Inf after forward_backward_func returns.

    This function intentionally does not raise immediately.
    It records state and lets the commit barrier decide before weight update.
    """

    def wrapped_forward_backward_func(*args, **kwargs):
        losses_reduced = forward_backward_func(*args, **kwargs)

        for _, value in _extract_loss_items(losses_reduced):
            if _is_non_finite(value):
                reason, scalar = _classify(
                    value,
                    RetryReason.LOSS_NAN,
                    RetryReason.LOSS_INF,
                )
                step_health.mark_loss_invalid(reason, scalar)
                break

        return losses_reduced

    return wrapped_forward_backward_func


def _found_inf_to_bool(found_inf: Any) -> bool:
    """Convert Megatron optimizer found_inf to a python bool.

    found_inf may be bool, int, float, tensor, or None.
    """

    if found_inf is None:
        return False

    try:
        if torch.is_tensor(found_inf):
            if found_inf.numel() == 0:
                return False
            return bool(found_inf.detach().float().max().item() != 0.0)

        return bool(found_inf)
    except Exception:
        return False


class PreCommitGuard:
    """Patch optimizer methods to catch NaN/Inf before parameter commit."""

    def __init__(self, optimizer, failure_sync, step_health: StepHealth):
        self.optimizer = optimizer
        self.failure_sync = failure_sync
        self.step_health = step_health

    def wrap_prepare_grads(self, original_prepare_grads):
        def wrapped_prepare_grads(*args, **kwargs):
            found_inf = original_prepare_grads(*args, **kwargs)

            # found_inf may early-return in optimizer.step() and never reach
            # step_with_ready_grads(). Therefore every rank must enter this
            # recovery sync point, even when local found_inf is False.
            if _found_inf_to_bool(found_inf):
                self.step_health.mark_overflow()

            self._barrier_and_raise("after_prepare_grads")

            return found_inf

        return wrapped_prepare_grads

    def wrap_get_grad_norm(self, original_get_grad_norm):
        def wrapped_get_grad_norm(*args, **kwargs):
            grad_norm = original_get_grad_norm(*args, **kwargs)
            self._record_grad_norm(grad_norm)
            return grad_norm

        return wrapped_get_grad_norm

    def wrap_clip_grad_norm(self, original_clip_grad_norm):
        def wrapped_clip_grad_norm(*args, **kwargs):
            grad_norm = original_clip_grad_norm(*args, **kwargs)
            self._record_grad_norm(grad_norm)
            return grad_norm

        return wrapped_clip_grad_norm

    def wrap_step_with_ready_grads(self, original_step_with_ready_grads):
        def wrapped_step_with_ready_grads(*args, **kwargs):
            # This is the real optimizer commit point.
            # Model weight, FP32 master weight, Adam state, optimizer step count
            # may change after this call.
            self._barrier_and_raise("before_step_with_ready_grads")
            return original_step_with_ready_grads(*args, **kwargs)

        return wrapped_step_with_ready_grads

    def _record_grad_norm(self, grad_norm):
        if _is_non_finite(grad_norm):
            reason, scalar = _classify(
                grad_norm,
                RetryReason.GRAD_NAN,
                RetryReason.GRAD_INF,
            )
            self.step_health.mark_grad_norm_invalid(reason, scalar)

    def _barrier_and_raise(self, barrier_name: str):
        local_invalid = self.step_health.invalid
        local_reason = self.step_health.reason if local_invalid else RetryReason.NONE

        # Every rank must participate before any rank raises.
        global_invalid, global_reason = self.failure_sync.sync_failure(
            local_invalid,
            local_reason,
        )

        if not global_invalid:
            return

        raise RetryableStepError(
            reason=global_reason,
            loss_value=self.step_health.loss_value,
            grad_norm_value=self.step_health.grad_norm_value,
            message=(
                f"HiF8 pre-commit recovery triggered at {barrier_name}. "
                f"local_invalid={local_invalid}, "
                f"local_reason={local_reason.name}, "
                f"global_reason={global_reason.name}, "
                f"loss={self.step_health.loss_value}, "
                f"grad_norm={self.step_health.grad_norm_value}"
            ),
        )


@contextmanager
def patch_optimizer_pre_commit_methods(optimizer, failure_sync, step_health: StepHealth):
    """Temporarily patch optimizer methods during one train_step call.

    Do not patch optimizer.step().
    Do not patch sub optimizers.
    Only patch top-level optimizer methods.
    """

    guard = PreCommitGuard(
        optimizer=optimizer,
        failure_sync=failure_sync,
        step_health=step_health,
    )

    originals: List[Tuple[Any, str, Any]] = []

    def patch_method(obj, name, wrapper_factory):
        if not hasattr(obj, name):
            return False

        original = getattr(obj, name)
        wrapped = wrapper_factory(original)
        setattr(obj, name, wrapped)
        originals.append((obj, name, original))
        return True

    patch_method(optimizer, "prepare_grads", guard.wrap_prepare_grads)
    patch_method(optimizer, "get_grad_norm", guard.wrap_get_grad_norm)
    patch_method(optimizer, "clip_grad_norm", guard.wrap_clip_grad_norm)

    has_commit_barrier = patch_method(
        optimizer,
        "step_with_ready_grads",
        guard.wrap_step_with_ready_grads,
    )

    if not has_commit_barrier:
        raise RuntimeError(
            "HiF8 pre-commit recovery requires optimizer.step_with_ready_grads(). "
            f"Unsupported optimizer type: {type(optimizer).__name__}"
        )

    try:
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
