# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for TeacherModelManager - manages teacher model lifecycle for QAD training."""

import pytest
from unittest.mock import MagicMock, patch

from mindspeed.core.distill.teacher_model_manager import TeacherModelManager


class TestTeacherModelManagerInit:
    """Test TeacherModelManager initialization and state."""

    def test_init_stores_config_and_student(self, config):
        """Manager stores config/student, starts unloaded; is_loaded needs flag + model."""
        student = MagicMock()
        mgr = TeacherModelManager(config, student)
        assert mgr.config is config
        assert mgr.student_model is student
        assert mgr.teacher_model is None
        assert mgr.is_loaded() is False

    def test_is_loaded_requires_model(self, config):
        """is_loaded returns True only when teacher_model is set."""
        mgr = TeacherModelManager(config)
        mgr._is_loaded = True
        assert mgr.is_loaded() is False  # flag set but model is None
        mgr.teacher_model = MagicMock()
        assert mgr.is_loaded() is True


class TestTeacherModelManagerLoad:
    """Test load_teacher and checkpoint loading."""

    def test_load_returns_existing_if_loaded(self, manager):
        """load_teacher returns existing model if already loaded."""
        mock_teacher = MagicMock()
        manager.teacher_model = mock_teacher
        manager._is_loaded = True
        result = manager.load_teacher()
        assert result is mock_teacher

    @patch.object(TeacherModelManager, '_load_checkpoint')
    @patch.object(TeacherModelManager, '_freeze_teacher')
    def test_load_creates_from_provider(self, mock_freeze, mock_load, config):
        """load_teacher creates model from provider, loads checkpoint, freezes."""
        manager = TeacherModelManager(config)
        mock_model = MagicMock()
        mock_provider = MagicMock(return_value=mock_model)

        result = manager.load_teacher(model_provider_func=mock_provider)

        assert result is mock_model
        mock_provider.assert_called_once()
        mock_load.assert_called_once()
        mock_freeze.assert_called_once()

    @pytest.mark.parametrize(
        "ckpt_behavior",
        [
            "no_checkpoint",  # _load_base_checkpoint returns (None, ...) → RuntimeError
            RuntimeError("checkpoint load failed"),  # exception propagates
        ],
    )
    def test_load_checkpoint_failure_raises(self, config, ckpt_behavior):
        """_load_checkpoint failure raises RuntimeError, not silently falling back."""
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()

        mock_ckpt_module = MagicMock()
        if isinstance(ckpt_behavior, BaseException):
            mock_ckpt_module._load_base_checkpoint.side_effect = ckpt_behavior
        else:
            # Return a 4-tuple with state_dict=None to trigger "No checkpoint found"
            mock_ckpt_module._load_base_checkpoint.return_value = (None, None, None, None)
        mock_utils_module = MagicMock()
        mock_utils_module.unwrap_model.return_value = manager.teacher_model

        with (
            patch("os.path.exists", return_value=True),
            patch.dict(
                'sys.modules',
                {
                    'megatron.training.checkpointing': mock_ckpt_module,
                    'megatron.training.utils': mock_utils_module,
                },
            ),
        ):
            with pytest.raises(RuntimeError):
                manager._load_checkpoint("/fake/path")


class TestTeacherModelManagerTpValidation:
    """Test _load_checkpoint TP-size validation (reviewer comment 3).

    The teacher checkpoint is loaded with rank0=False so each TP rank loads its
    own shard. If the checkpoint was saved with a different TP size, every rank
    loads the wrong weight partition. strict=False would silently accept this,
    producing meaningless KL targets. The validation must panic on mismatch.
    """

    def _make_state_dict(self, saved_tp_size):
        """Build a checkpoint state_dict with the given saved TP size."""
        saved_args = MagicMock()
        saved_args.tensor_model_parallel_size = saved_tp_size
        return {'model': {}, 'args': saved_args}

    def _setup_mocks(self, state_dict, current_tp_size):
        """Return (mock_ckpt_module, mock_utils_module, get_args_patch) for a run."""
        mock_ckpt_module = MagicMock()
        mock_ckpt_module._load_base_checkpoint.return_value = (
            state_dict,
            "ckpt_name",
            False,
            None,
        )
        mock_utils_module = MagicMock()
        unwrapped_model = MagicMock()
        # load_state_dict returns (missing_keys, unexpected_keys)
        unwrapped_model.load_state_dict.return_value = ([], [])
        mock_utils_module.unwrap_model.return_value = unwrapped_model

        mock_args = MagicMock()
        mock_args.tensor_model_parallel_size = current_tp_size
        return mock_ckpt_module, mock_utils_module, mock_args

    def test_tp_mismatch_raises(self, config):
        """Checkpoint saved with TP=1 but training with TP=8 -> RuntimeError."""
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()

        state_dict = self._make_state_dict(saved_tp_size=1)
        mock_ckpt, mock_utils, mock_args = self._setup_mocks(state_dict, current_tp_size=8)

        with (
            patch("os.path.exists", return_value=True),
            patch("mindspeed.core.distill.teacher_model_manager.get_args", return_value=mock_args),
            patch.dict(
                'sys.modules',
                {
                    'megatron.training.checkpointing': mock_ckpt,
                    'megatron.training.utils': mock_utils,
                },
            ),
        ):
            with pytest.raises(RuntimeError, match="TP size mismatch"):
                manager._load_checkpoint("/fake/path")

    def test_tp_match_proceeds_to_load(self, config):
        """Checkpoint saved with TP=8 and training with TP=8 -> no error."""
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()

        state_dict = self._make_state_dict(saved_tp_size=8)
        mock_ckpt, mock_utils, mock_args = self._setup_mocks(state_dict, current_tp_size=8)

        with (
            patch("os.path.exists", return_value=True),
            patch("mindspeed.core.distill.teacher_model_manager.get_args", return_value=mock_args),
            patch.dict(
                'sys.modules',
                {
                    'megatron.training.checkpointing': mock_ckpt,
                    'megatron.training.utils': mock_utils,
                },
            ),
        ):
            # Should not raise; load_state_dict is called on the unwrapped model.
            manager._load_checkpoint("/fake/path")

    def test_missing_saved_args_is_tolerated(self, config):
        """Checkpoint without 'args' key (old format) -> no TP check, no error."""
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()

        # state_dict has no 'args' key
        state_dict = {'model': {}}
        mock_ckpt, mock_utils, mock_args = self._setup_mocks(state_dict, current_tp_size=8)

        with (
            patch("os.path.exists", return_value=True),
            patch("mindspeed.core.distill.teacher_model_manager.get_args", return_value=mock_args),
            patch.dict(
                'sys.modules',
                {
                    'megatron.training.checkpointing': mock_ckpt,
                    'megatron.training.utils': mock_utils,
                },
            ),
        ):
            manager._load_checkpoint("/fake/path")

    def test_dict_format_saved_args_mismatch_raises(self, config):
        """Checkpoint whose 'args' is a plain dict (not Namespace) with TP
        mismatch -> RuntimeError. Guards against the check being silently
        skipped for dict-format args (getattr on a dict returns the default).
        """
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()

        # saved_args is a dict (some checkpoint converters save args as dict)
        saved_args = {'tensor_model_parallel_size': 1}
        state_dict = {'model': {}, 'args': saved_args}
        mock_ckpt, mock_utils, mock_args = self._setup_mocks(state_dict, current_tp_size=8)

        with (
            patch("os.path.exists", return_value=True),
            patch("mindspeed.core.distill.teacher_model_manager.get_args", return_value=mock_args),
            patch.dict(
                'sys.modules',
                {
                    'megatron.training.checkpointing': mock_ckpt,
                    'megatron.training.utils': mock_utils,
                },
            ),
        ):
            with pytest.raises(RuntimeError, match="TP size mismatch"):
                manager._load_checkpoint("/fake/path")

    def test_dict_format_saved_args_match_proceeds(self, config):
        """Checkpoint whose 'args' is a plain dict with matching TP -> no error."""
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()

        saved_args = {'tensor_model_parallel_size': 8}
        state_dict = {'model': {}, 'args': saved_args}
        mock_ckpt, mock_utils, mock_args = self._setup_mocks(state_dict, current_tp_size=8)

        with (
            patch("os.path.exists", return_value=True),
            patch("mindspeed.core.distill.teacher_model_manager.get_args", return_value=mock_args),
            patch.dict(
                'sys.modules',
                {
                    'megatron.training.checkpointing': mock_ckpt,
                    'megatron.training.utils': mock_utils,
                },
            ),
        ):
            manager._load_checkpoint("/fake/path")


class TestTeacherModelManagerGetLogits:
    """Test get_logits and memory usage."""

    def test_get_logits_returns_output(self, config):
        """get_logits returns logits when model is loaded."""
        manager = TeacherModelManager(config)
        mock_teacher = MagicMock()
        mock_teacher.to.return_value = mock_teacher
        mock_output = MagicMock()
        mock_teacher.return_value = mock_output
        manager.teacher_model = mock_teacher
        manager._is_loaded = True

        result = manager.get_logits(MagicMock(), MagicMock(), MagicMock())
        assert result is mock_output

    def test_get_memory_usage(self, config):
        """get_memory_usage returns dict with expected keys when loaded."""
        manager = TeacherModelManager(config)
        manager.teacher_model = MagicMock()
        manager._is_loaded = True
        usage = manager.get_memory_usage()
        assert isinstance(usage, dict)
        assert usage["loaded"] is True
        assert "total_params" in usage
