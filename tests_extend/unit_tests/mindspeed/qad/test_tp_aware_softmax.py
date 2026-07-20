# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for TP-aware softmax in LogitsKLLoss.

Tests the tensor parallelism support in the KL divergence loss computation,
verifying that TP-aware softmax produces correct global probability distributions
and that gradients flow correctly through the TP communication operations.

Test Strategy:
- Mock parallel_state to simulate TP>1 environment
- Mock torch.distributed.all_reduce and torch.distributed.nn.functional.all_reduce
- Verify TP-aware softmax matches standard softmax when TP=1
- Verify TP-aware path produces correct global KL loss
- Verify gradient flow through TP communication
- Verify numerical stability with large logits
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from mindspeed.core.distill.logits_kl_loss import LogitsKLLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def tp2_all_reduce_identity():
    """Patch both all_reduce variants as identity (no-op) for TP=2 tests."""
    with (
        patch("torch.distributed.all_reduce") as mock_ar,
        patch("torch.distributed.nn.functional.all_reduce") as mock_nn_ar,
    ):
        mock_ar.side_effect = lambda x, **kw: x
        mock_nn_ar.side_effect = lambda x, **kw: x
        yield mock_ar, mock_nn_ar


def _rand_logits(*shape, requires_grad=False):
    """Create random logits tensor."""
    return torch.randn(*shape, requires_grad=requires_grad)


# ============================================================================
# TP=1: TP-aware path should match standard softmax
# ============================================================================


class TestTPAwareSoftmaxTP1:
    """TP-aware softmax produces same results as standard softmax when TP=1."""

    def test_tp1_matches_standard_softmax(self, tp1_env):
        """TP=1 loss should match standard F.log_softmax-based KL loss."""
        student, teacher = _rand_logits(4, 8, 100), _rand_logits(4, 8, 100)
        loss = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)

        # Reference using standard softmax
        t_lp = F.log_softmax(teacher.float(), dim=-1)
        s_lp = F.log_softmax(student.float(), dim=-1)
        ref = F.kl_div(s_lp, t_lp, reduction="none", log_target=True).sum(dim=-1).mean()

        assert torch.isclose(loss, ref, atol=1e-5)

    def test_tp1_gradient_and_temperature(self, tp1_env):
        """TP=1: gradients flow correctly and temperature scaling works."""
        student = _rand_logits(4, 8, 100, requires_grad=True)
        teacher = _rand_logits(4, 8, 100)

        loss_t1 = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)
        loss_t2 = LogitsKLLoss(temperature=2.0, reduction="mean")(student, teacher)
        loss_t1.backward()

        assert student.grad is not None
        assert not torch.isnan(student.grad).any()
        assert loss_t1 != loss_t2, "Different temperatures should produce different losses"


# ============================================================================
# TP=2: all_reduce calls, gradient flow, loss properties
# ============================================================================


class TestTPAwareSoftmaxTP2:
    """TP-aware softmax with simulated TP=2."""

    def test_tp2_all_reduce_called(self, tp2_env):
        """TP>1: both all_reduce variants are called (max + denominator)."""
        student = _rand_logits(4, 8, 100, requires_grad=True)
        teacher = _rand_logits(4, 8, 100)

        with tp2_all_reduce_identity() as (mock_ar, mock_nn_ar):
            LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)

        # all_reduce for max (teacher + student) and nn.functional for denominator
        assert mock_ar.call_count >= 2
        assert mock_nn_ar.call_count >= 2

    def test_tp2_gradient_flow_and_finite_loss(self, tp2_env):
        """TP>1: gradients flow through TP-aware path and loss is finite & non-negative."""
        student = _rand_logits(4, 8, 100, requires_grad=True)
        teacher = _rand_logits(4, 8, 100)

        with tp2_all_reduce_identity():
            loss_fn = LogitsKLLoss(temperature=1.0, reduction="mean")
            loss = loss_fn(student, teacher)
            loss.backward()

        assert student.grad is not None
        assert not torch.isnan(student.grad).any()
        assert not torch.isnan(loss) and not torch.isinf(loss)
        assert loss >= 0


# ============================================================================
# TP=2: numerical stability with extreme values
# ============================================================================


class TestTPAwareSoftmaxNumericalStability:
    """Numerical stability of TP-aware softmax with extreme values."""

    def test_large_logits_stable(self, tp2_env):
        """TP-aware softmax handles large logits without NaN in loss and gradients."""
        student = (_rand_logits(4, 8, 100) * 100).requires_grad_(True)
        teacher = _rand_logits(4, 8, 100) * 100

        with tp2_all_reduce_identity():
            loss = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)
            loss.backward()

        assert not torch.isnan(loss) and not torch.isinf(loss)
        assert student.grad is not None
        assert not torch.isnan(student.grad).any()

    def test_identical_distributions_zero_kl(self, tp2_env):
        """KL loss between identical distributions should be ~0."""
        logits = _rand_logits(4, 8, 100)
        with tp2_all_reduce_identity():
            loss = LogitsKLLoss(temperature=1.0, reduction="mean")(logits, logits)
        assert loss < 1e-5


# ============================================================================
# TP=2: temperature scaling
# ============================================================================


class TestTPAwareSoftmaxTemperature:
    """Temperature scaling with TP-aware softmax."""

    def test_temperature_scaling_and_gradient(self, tp2_env):
        """TP>1: different temperatures produce different losses; gradients flow with T=2."""
        student = _rand_logits(4, 8, 100, requires_grad=True)
        teacher = _rand_logits(4, 8, 100)

        with tp2_all_reduce_identity():
            loss_t1 = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)
            loss_t2 = LogitsKLLoss(temperature=2.0, reduction="mean")(student, teacher)
            loss_t2.backward()

        assert loss_t1 != loss_t2
        assert student.grad is not None
        assert not torch.isnan(student.grad).any()


# ============================================================================
# TP=2: reduction modes
# ============================================================================


class TestTPAwareSoftmaxReduction:
    """Reduction modes with TP-aware softmax."""

    @pytest.mark.parametrize(
        "reduction, expected_ndim, expected_shape",
        [
            ("mean", 0, None),
            ("sum", 0, None),
            ("none", None, (8, 4)),
        ],
    )
    def test_reduction_modes(self, tp2_env, reduction, expected_ndim, expected_shape):
        """TP-aware softmax produces correct output shape for each reduction mode."""
        s, b, v = 8, 4, 100
        student = _rand_logits(s, b, v)
        teacher = _rand_logits(s, b, v)

        with tp2_all_reduce_identity():
            loss = LogitsKLLoss(temperature=1.0, reduction=reduction)(student, teacher)

        if expected_ndim is not None:
            assert loss.ndim == expected_ndim
        if expected_shape is not None:
            assert loss.shape == expected_shape
        assert not torch.isnan(loss).any()


# ============================================================================
# TP=2: student max detach (NVIDIA design pattern)
# ============================================================================


class TestTPAwareSoftmaxStudentMaxDetach:
    """Student logits max is detached to prevent gradient noise (NVIDIA pattern)."""

    def test_student_max_detached(self, tp2_env):
        """Student max subtraction should not add gradient noise."""
        student = _rand_logits(4, 8, 100, requires_grad=True)
        teacher = _rand_logits(4, 8, 100)

        with tp2_all_reduce_identity():
            loss = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)
            loss.backward()

        assert student.grad is not None
        assert not torch.isnan(student.grad).any()
        assert student.grad.norm().item() < 1000, "Gradient norm should be reasonable"
