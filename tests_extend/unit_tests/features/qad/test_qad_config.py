"""Unit tests for QADConfig dataclass — fields, defaults, and validation."""

import pytest

from mindspeed.core.distill.config import QADConfig


class TestQADConfig:
    """Test QADConfig field defaults, custom assignment, and validation."""

    def test_defaults_and_custom(self):
        """Default constructor and full custom assignment both work correctly."""
        # Defaults
        d = QADConfig()
        assert d.enabled is False
        assert d.teacher_checkpoint_path == ""
        assert d.kl_temperature == 1.0
        assert d.kl_loss_weight == 1.0
        assert d.kl_reduction == "mean"
        assert not d.extra_config

        # Custom
        d = QADConfig(
            enabled=True,
            teacher_checkpoint_path="/path/to/ckpt",
            kl_temperature=2.0,
            kl_loss_weight=0.5,
            kl_reduction="sum",
            extra_config={"key": "value"},
        )
        assert d.enabled is True
        assert d.teacher_checkpoint_path == "/path/to/ckpt"
        assert d.kl_temperature == 2.0
        assert d.kl_loss_weight == 0.5
        assert d.kl_reduction == "sum"
        assert d.extra_config == {"key": "value"}

    def test_valid_config_passes_validation(self):
        """A fully-specified valid config passes validate() without raising."""
        d = QADConfig(
            enabled=True,
            teacher_checkpoint_path="/path/to/ckpt",
            kl_temperature=2.0,
            kl_loss_weight=0.5,
            kl_reduction="sum",
        )
        d.validate()  # should not raise

    @pytest.mark.parametrize(
        "kl_weight, expected_kl",
        [
            (1.0, 1.0),
            (0.0, 0.0),
            (0.5, 0.5),
        ],
    )
    def test_total_loss_weight(self, kl_weight, expected_kl):
        """total_loss_weight returns (0.0, kl_weight) — CE weight is always 0 in KL-only mode."""
        d = QADConfig(kl_loss_weight=kl_weight)
        ce, kl = d.total_loss_weight()
        assert ce == 0.0
        assert kl == expected_kl
