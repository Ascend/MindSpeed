# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for QADQuantEngineFeature.pre_validate_args and is_need_apply.

Tests that argument validation raises AssertionError on invalid input
(fail-fast) instead of silently returning False.
"""

from argparse import Namespace

import pytest

# QADQuantEngineFeature is loaded by conftest.py via importlib.util to avoid
# importing the full mindspeed.features_manager package (which pulls in
# torch_npu and other heavy dependencies).


def _make_args(**kwargs):
    """Create a Namespace with QAD defaults, overridden by kwargs."""
    defaults = dict(
        qad_enable=False,
        qad_teacher_load='',
        kl_temperature=1.0,
        kl_loss_weight=1.0,
        kl_loss_reduction='mean',
        num_experts=None,
        pipeline_model_parallel_size=1,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestPreValidateArgs:
    """Test pre_validate_args: valid args pass, invalid args raise AssertionError."""

    @pytest.mark.parametrize(
        "label, kwargs",
        [
            ("disabled_skips_validation", {"qad_enable": False, "qad_teacher_load": ""}),
            (
                "valid_full_args",
                {
                    "qad_enable": True,
                    "qad_teacher_load": "/path/to/teacher",
                    "kl_temperature": 2.0,
                    "kl_loss_weight": 0.5,
                    "kl_loss_reduction": "sum",
                },
            ),
            ("zero_kl_weight_valid", {"qad_enable": True, "qad_teacher_load": "/path", "kl_loss_weight": 0.0}),
        ],
    )
    def test_valid_args_pass(self, feature, label, kwargs):
        """Valid argument combinations should not raise."""
        feature.pre_validate_args(_make_args(**kwargs))

    @pytest.mark.parametrize(
        "label, kwargs, match",
        [
            ("missing_teacher_path", {"qad_enable": True, "qad_teacher_load": ""}, "qad-teacher-load"),
            (
                "negative_temperature",
                {"qad_enable": True, "qad_teacher_load": "/path", "kl_temperature": -1.0},
                "kl-temperature",
            ),
            (
                "zero_temperature",
                {"qad_enable": True, "qad_teacher_load": "/path", "kl_temperature": 0.0},
                "kl-temperature",
            ),
            (
                "negative_loss_weight",
                {"qad_enable": True, "qad_teacher_load": "/path", "kl_loss_weight": -0.5},
                "kl-loss-weight",
            ),
            (
                "invalid_reduction",
                {"qad_enable": True, "qad_teacher_load": "/path", "kl_loss_reduction": "invalid"},
                "kl-loss-reduction",
            ),
            (
                "moe_not_supported",
                {"qad_enable": True, "qad_teacher_load": "/path", "num_experts": 8},
                "MoE",
            ),
            (
                "pp_not_supported",
                {"qad_enable": True, "qad_teacher_load": "/path", "pipeline_model_parallel_size": 4},
                "pipeline parallelism",
            ),
        ],
    )
    def test_invalid_args_raise(self, feature, label, kwargs, match):
        """Invalid argument combinations should raise AssertionError (fail-fast)."""
        with pytest.raises(AssertionError, match=match):
            feature.pre_validate_args(_make_args(**kwargs))


class TestIsNeedApply:
    """Test the is_need_apply gating logic."""

    @pytest.mark.parametrize(
        "args, expected",
        [
            (_make_args(qad_enable=True), True),
            (_make_args(qad_enable=False), False),
            (Namespace(), False),  # missing qad_enable attribute defaults to False
        ],
    )
    def test_is_need_apply(self, feature, args, expected):
        assert feature.is_need_apply(args) is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
