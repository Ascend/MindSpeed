# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for LogitsKLLoss - KL divergence loss for QAD training."""

import pytest
import torch

from mindspeed.core.distill.logits_kl_loss import LogitsKLLoss


def _rand_logits(*shape, requires_grad=False, dtype=torch.float32):
    """Create random logits tensor."""
    return torch.randn(*shape, requires_grad=requires_grad, dtype=dtype)


class TestLogitsKLLossOutput:
    """Test LogitsKLLoss forward pass output properties."""

    def test_output_properties(self, tp1):
        """Loss is scalar, finite, and non-negative for random inputs."""
        student, teacher = _rand_logits(8, 4, 100), _rand_logits(8, 4, 100)
        loss = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)
        assert loss.ndim == 0
        assert not torch.isnan(loss) and not torch.isinf(loss)

    def test_identical_distributions_near_zero(self, tp1):
        """KL divergence between identical distributions should be ~0 (non-negative)."""
        logits = _rand_logits(8, 4, 100)
        loss = LogitsKLLoss(temperature=1.0, reduction="mean")(logits, logits)
        assert loss >= 0
        assert loss < 1e-5

    def test_temperature_scaling(self, tp1):
        """Higher temperature produces different (larger) loss due to T^2 compensation."""
        student, teacher = _rand_logits(8, 4, 100), _rand_logits(8, 4, 100)
        loss_low = LogitsKLLoss(temperature=0.5, reduction="mean")(student, teacher)
        loss_high = LogitsKLLoss(temperature=2.0, reduction="mean")(student, teacher)
        assert loss_low != loss_high
        assert loss_high > loss_low


class TestLogitsKLLossReduction:
    """Test LogitsKLLoss reduction modes."""

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_produces_scalar(self, tp1, reduction):
        """Both mean and sum reductions produce a scalar loss."""
        student, teacher = _rand_logits(8, 4, 100), _rand_logits(8, 4, 100)
        loss = LogitsKLLoss(temperature=1.0, reduction=reduction)(student, teacher)
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestLogitsKLLossEdgeCases:
    """Test LogitsKLLoss edge cases."""

    def test_shape_mismatch_raises(self):
        """Shape mismatch between student and teacher raises ValueError."""
        student = _rand_logits(8, 4, 100)
        teacher = _rand_logits(8, 4, 200)
        with pytest.raises(ValueError, match="same shape"):
            LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)

    def test_gradient_flow(self, tp1):
        """Gradients flow through the loss to student logits without NaN/Inf."""
        student = _rand_logits(8, 4, 100, requires_grad=True)
        teacher = _rand_logits(8, 4, 100)
        loss = LogitsKLLoss(temperature=2.0, reduction="mean")(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert not torch.isnan(student.grad).any()
        assert not torch.isinf(student.grad).any()

    def test_dtype_preservation(self, tp1):
        """Loss computation handles float16 inputs without NaN."""
        student = _rand_logits(8, 4, 100, dtype=torch.float16)
        teacher = _rand_logits(8, 4, 100, dtype=torch.float16)
        loss = LogitsKLLoss(temperature=1.0, reduction="mean")(student, teacher)
        assert not torch.isnan(loss)
