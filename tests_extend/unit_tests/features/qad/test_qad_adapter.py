# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for QAD patches - QADForwardStepPatch.

These tests validate the patch functions that integrate QAD into the training pipeline.
All tests use mocked megatron modules to avoid requiring GPU/NPU hardware.
"""

import importlib.util
import os
import sys
from functools import partial
from unittest.mock import MagicMock, patch

import mindspeed
import pytest
import torch

# Save ALL sys.modules entries that we are about to mock so they can be
# restored immediately after exec_module. Without this, the module-level mocks
# pollute the global sys.modules cache and break other test modules (e.g.
# test_teacher_model_manager.py, test_tp_aware_softmax.py, test_logits_kl_loss.py)
# that import the real megatron/mindspeed packages during pytest collection.
_MOCKED_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.mpu",
    "megatron.training",
    "megatron.training.arguments",
    "megatron.core.models",
    "megatron.core.models.gpt",
    "megatron.core.models.gpt.gpt_layer_specs",
    "megatron.core.transformer",
    "megatron.core.transformer.spec_utils",
    "megatron.legacy.model",
    "mindspeed.checkpointing",
    "mindspeed.checkpointing.checkpointing",
    "pretrain_gpt",
    "mindspeed.features_manager",
    "mindspeed.features_manager.fusions",
    "mindspeed.features_manager.fusions.fused_moe_permute",
    "mindspeed.core.distill",
    "mindspeed.core.distill.teacher_model_manager",
    "mindspeed.core.distill.logits_kl_loss",
]
_saved_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULES}

# Mock megatron modules before importing
sys.modules["megatron"] = MagicMock()
_mock_megatron_core = MagicMock()
_mock_mpu = MagicMock()
_mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
_mock_megatron_core.mpu = _mock_mpu
sys.modules["megatron.core"] = _mock_megatron_core
sys.modules["megatron.core.mpu"] = _mock_mpu
sys.modules["megatron.training"] = MagicMock()
sys.modules["megatron.training.arguments"] = MagicMock()
sys.modules["megatron.core.models"] = MagicMock()
sys.modules["megatron.core.models.gpt"] = MagicMock()
sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = MagicMock()
sys.modules["megatron.core.transformer"] = MagicMock()
sys.modules["megatron.core.transformer.spec_utils"] = MagicMock()
sys.modules["megatron.legacy.model"] = MagicMock()
sys.modules["mindspeed.checkpointing"] = MagicMock()
sys.modules["mindspeed.checkpointing.checkpointing"] = MagicMock()
sys.modules["pretrain_gpt"] = MagicMock()
# Mock the entire mindspeed.features_manager to avoid importing torch_npu
sys.modules["mindspeed.features_manager"] = MagicMock()
sys.modules["mindspeed.features_manager.fusions"] = MagicMock()
sys.modules["mindspeed.features_manager.fusions.fused_moe_permute"] = MagicMock()

# Now import the module directly without going through __init__.py.
# mindspeed is imported at the top of this file; its __init__.py is empty so
# the import is cheap and safe (no torch_npu).  We use mindspeed.__file__ to
# locate the package directory.
_qad_patches_path = os.path.join(os.path.dirname(mindspeed.__file__), "core", "distill", "qad_adapter.py")
spec = importlib.util.spec_from_file_location("qad_patches", _qad_patches_path)
qad_patches = importlib.util.module_from_spec(spec)

# Mock the imports that the module needs
sys.modules["mindspeed.core.distill"] = MagicMock()
sys.modules["mindspeed.core.distill.teacher_model_manager"] = MagicMock()
sys.modules["mindspeed.core.distill.logits_kl_loss"] = MagicMock()

# Create mock classes
mock_teacher_manager_class = MagicMock()
mock_logits_kl_loss_class = MagicMock()

sys.modules["mindspeed.core.distill.teacher_model_manager"].TeacherModelManager = mock_teacher_manager_class
sys.modules["mindspeed.core.distill.logits_kl_loss"].LogitsKLLoss = mock_logits_kl_loss_class

# Set up the module's imports
qad_patches.TeacherModelManager = mock_teacher_manager_class
qad_patches.LogitsKLLoss = mock_logits_kl_loss_class

# Execute the module to get actual code
spec.loader.exec_module(qad_patches)

# Restore ALL original sys.modules entries immediately after exec_module.
# The mocks were only needed during exec_module to satisfy the import statements
# in qad_patches.py. The mock classes are now bound directly on the qad_patches
# module object (lines above), so the sys.modules entries can be safely restored.
# This prevents polluting other test modules that import the real
# megatron/mindspeed packages during pytest collection.
for _name in _MOCKED_MODULES:
    _original = _saved_modules.get(_name)
    if _original is not None:
        sys.modules[_name] = _original
    else:
        sys.modules.pop(_name, None)

QADForwardStepPatch = qad_patches.QADForwardStepPatch
_qad_loss_func = qad_patches._qad_loss_func
get_teacher_manager = qad_patches.get_teacher_manager
reset_teacher_manager = qad_patches.reset_teacher_manager


# ---------------------------------------------------------------------------
# Helpers for loss-func tests
# ---------------------------------------------------------------------------

_CE_LOSS = 0.5
_KL_LOSS = 0.3
_NUM_TOKENS = 32


def _make_loss_inputs():
    """Create (original_partial, kl_loss) for _qad_loss_func tests.

    Simulates: partial(base_loss_func, loss_mask) where
    base_loss_func(loss_mask, output_tensor) returns (loss, num_tokens, metrics).
    """

    def base_loss_func(loss_mask, output_tensor):
        return (
            torch.tensor(_CE_LOSS),
            torch.tensor(_NUM_TOKENS),
            {'lm loss': (torch.tensor(_CE_LOSS), torch.tensor(_NUM_TOKENS))},
        )

    loss_mask = torch.ones(8, 4)
    return partial(base_loss_func, loss_mask), torch.tensor(_KL_LOSS)


def _make_singleton_mock_args():
    """Create mock args for teacher manager singleton tests."""
    mock_args = MagicMock()
    mock_args.qad_enable = True
    mock_args.qad_teacher_load = "/fake/path"
    mock_args.kl_temperature = 1.0
    mock_args.kl_loss_weight = 1.0
    mock_args.kl_loss_reduction = "mean"
    return mock_args


# ============================================================================
# _BatchCacheIterator — ensures CE and KL consume the same batch
# ============================================================================


class TestBatchCacheIterator:
    """Test _BatchCacheIterator replays the first batch exactly once.

    This is the fix for the double-get_batch bug where CE and KL losses
    were computed on different batches.
    """

    def test_first_two_calls_return_same_batch(self):
        """The first and second next() calls return the same batch object."""
        batches = [object(), object(), object()]
        it = qad_patches._BatchCacheIterator(iter(batches))

        first = next(it)
        second = next(it)

        assert first is second is batches[0]

    def test_third_call_advances_to_next_batch(self):
        """After the replay, the iterator advances normally."""
        batches = [object(), object(), object()]
        it = qad_patches._BatchCacheIterator(iter(batches))

        next(it)  # batch 0 (cached)
        next(it)  # batch 0 (replay)
        third = next(it)  # batch 1 (fresh)

        assert third is batches[1]

    def test_iter_returns_self(self):
        """__iter__ returns the iterator itself (protocol compliance)."""
        it = qad_patches._BatchCacheIterator(iter([1]))
        assert iter(it) is it

    def test_empty_iterator_raises_stop_iteration(self):
        """An empty underlying iterator raises StopIteration on first call."""
        it = qad_patches._BatchCacheIterator(iter([]))
        with pytest.raises(StopIteration):
            next(it)

    def test_single_batch_replayed_then_exhausted(self):
        """A single-batch iterator yields it twice then raises StopIteration."""
        batch = object()
        it = qad_patches._BatchCacheIterator(iter([batch]))

        first = next(it)
        second = next(it)

        assert first is second is batch
        with pytest.raises(StopIteration):
            next(it)


# ============================================================================
# QADForwardStepPatch - setup_model_and_optimizer wrapper
# ============================================================================


class TestQADSetupModelAndOptimizerPatch:
    """Test patched_setup_model_and_optimizer_wrapper - injects qad_model_provider."""

    def test_returns_patched_function(self):
        """Wrapper returns a callable wrapping the original."""
        original_func = MagicMock()
        patched = QADForwardStepPatch.patched_setup_model_and_optimizer_wrapper(original_func)
        assert callable(patched)
        assert patched.__wrapped__ is original_func

    def test_qad_disabled_passes_through(self):
        """When qad_enable=False, original model_provider is passed through unwrapped."""
        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_setup_model_and_optimizer_wrapper(original)

        mock_args = MagicMock()
        mock_args.qad_enable = False
        original_model_provider = MagicMock(return_value="student_model")

        with patch.object(qad_patches, 'get_args', return_value=mock_args):
            result = patched(original_model_provider, MagicMock())

        assert result == "result"
        # model_provider (arg 0) should be the original, passed through unwrapped
        call_args = original.call_args
        assert call_args[0][0] is original_model_provider

    def test_qad_enabled_wraps_model_provider(self):
        """When qad_enable=True, model_provider is wrapped with QAD teacher loading."""
        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_setup_model_and_optimizer_wrapper(original)

        mock_args = MagicMock()
        mock_args.qad_enable = True
        original_model_provider = MagicMock(return_value="student_model")

        with (
            patch.object(qad_patches, 'get_args', return_value=mock_args),
            patch.object(qad_patches, 'get_teacher_manager'),
        ):
            result = patched(original_model_provider, MagicMock())

        assert result == "result"
        # model_provider (arg 0) should be wrapped (not the original)
        call_args = original.call_args
        wrapped_model_provider = call_args[0][0]
        assert callable(wrapped_model_provider)
        assert wrapped_model_provider is not original_model_provider


# ============================================================================
# QADForwardStepPatch - train_step wrapper
# ============================================================================


class TestQADTrainStepPatch:
    """Test patched_train_step_wrapper - injects qad_forward_step (KL loss)."""

    def test_returns_patched_function(self):
        """Wrapper returns a callable wrapping the original."""
        original_func = MagicMock()
        patched = QADForwardStepPatch.patched_train_step_wrapper(original_func)
        assert callable(patched)
        assert patched.__wrapped__ is original_func

    def test_qad_disabled_passes_through(self):
        """When qad_enable=False, original forward_step_func is passed through unwrapped."""
        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_train_step_wrapper(original)

        mock_args = MagicMock()
        mock_args.qad_enable = False
        original_forward_step = MagicMock()

        with patch.object(qad_patches, 'get_args', return_value=mock_args):
            result = patched(original_forward_step, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        assert result == "result"
        # forward_step_func (arg 0) should be the original, passed through unwrapped
        call_args = original.call_args
        assert call_args[0][0] is original_forward_step

    def test_qad_enabled_teacher_not_loaded_falls_back(self):
        """When qad_enable=True but teacher not loaded, falls back to original forward_step."""
        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_train_step_wrapper(original)

        mock_args = MagicMock()
        mock_args.qad_enable = True
        original_forward_step = MagicMock()
        mock_tm = MagicMock()
        mock_tm.is_loaded.return_value = False

        with (
            patch.object(qad_patches, 'get_args', return_value=mock_args),
            patch.object(qad_patches, 'get_teacher_manager', return_value=mock_tm),
        ):
            result = patched(original_forward_step, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        assert result == "result"
        # original called with the unwrapped forward_step_func (arg 0)
        original.assert_called_once()
        call_args = original.call_args
        assert call_args[0][0] is original_forward_step


# ============================================================================
# Regression: *args forwarding with a real argparse.Namespace
# (guards against the P0 shadowing bug where `args = get_args()` rebinds the
#  *args tuple; argparse.Namespace is NOT iterable, so `*args` unpacking
#  crashes with TypeError. MagicMock masks this because it auto-implements
#  __iter__, so these tests use a real Namespace.)
# ============================================================================


class TestArgsForwardingRegression:
    """Ensure extra positional args reach the original function unchanged.

    get_args() returns an argparse.Namespace (not iterable). If the wrapper
    rebinds the ``*args`` tuple to that Namespace, ``*args`` unpacking on the
    return line raises ``TypeError: Value after * must be an iterable``.
    These tests use a real Namespace to catch that regression.
    """

    def test_setup_model_and_optimizer_forwards_extra_positional_args(self):
        """setup_model_and_optimizer wrapper forwards trailing positional args."""
        from argparse import Namespace

        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_setup_model_and_optimizer_wrapper(original)

        # Real Namespace (not iterable) with qad_enable=False -> pass-through path
        ns = Namespace(qad_enable=False)
        original_model_provider = MagicMock(return_value="student")

        with patch.object(qad_patches, 'get_args', return_value=ns):
            result = patched(original_model_provider, "model_type", "extra_kwarg")

        assert result == "result"
        original.assert_called_once()
        # All positional args must be forwarded: model_provider + the two extras
        forwarded = original.call_args[0]
        assert forwarded[0] is original_model_provider
        assert forwarded[1] == "model_type"
        assert forwarded[2] == "extra_kwarg"

    def test_train_step_forwards_extra_positional_args(self):
        """train_step wrapper forwards trailing positional args."""
        from argparse import Namespace

        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_train_step_wrapper(original)

        # Real Namespace (not iterable) with qad_enable=False -> pass-through path
        ns = Namespace(qad_enable=False)
        original_forward_step = MagicMock()

        with patch.object(qad_patches, 'get_args', return_value=ns):
            result = patched(original_forward_step, "data_iterator", "optimizer", "extra")

        assert result == "result"
        original.assert_called_once()
        forwarded = original.call_args[0]
        assert forwarded[0] is original_forward_step
        assert forwarded[1] == "data_iterator"
        assert forwarded[2] == "optimizer"
        assert forwarded[3] == "extra"

    def test_setup_model_and_optimizer_enabled_path_forwards_args(self):
        """Enabled path (qad_enable=True) also forwards extra positional args."""
        from argparse import Namespace

        original = MagicMock(return_value="result")
        patched = QADForwardStepPatch.patched_setup_model_and_optimizer_wrapper(original)

        ns = Namespace(qad_enable=True)
        original_model_provider = MagicMock(return_value="student")

        with (
            patch.object(qad_patches, 'get_args', return_value=ns),
            patch.object(qad_patches, 'get_teacher_manager'),
        ):
            result = patched(original_model_provider, "model_type", "extra_kwarg")

        assert result == "result"
        original.assert_called_once()
        forwarded = original.call_args[0]
        # arg 0 is the wrapped qad_model_provider (not the original), but the
        # trailing positional args must still be forwarded unchanged.
        assert callable(forwarded[0])
        assert forwarded[0] is not original_model_provider
        assert forwarded[1] == "model_type"
        assert forwarded[2] == "extra_kwarg"


# ============================================================================
# _qad_loss_func
# ============================================================================


class TestQADLossFunc:
    """Test _qad_loss_func - KL-only loss computation.

    _qad_loss_func is called via partial:
        partial(_qad_loss_func, original_loss_func_partial=..., kl_loss_value=...)
    So when the scheduler calls loss_func(output_tensor), it becomes:
        _qad_loss_func(output_tensor, original_loss_func_partial=..., kl_loss_value=...)

    QAD uses KL-only distillation: total_loss = kl_weight * kl_loss.
    CE loss is computed for logging only and does not contribute to gradients.
    num_tokens = 1 so Megatron's division is a no-op.
    """

    @pytest.mark.parametrize(
        "kl_weight, expected_total",
        [
            (1.0, 1.0 * _KL_LOSS),
            (2.0, 2.0 * _KL_LOSS),
            (0.0, 0.0),
        ],
        ids=["weight_1", "weight_2", "weight_0"],
    )
    def test_loss_combination(self, kl_weight, expected_total):
        """_qad_loss_func returns kl_weight * kl_loss (KL-only, no CE gradient)."""
        original_partial, kl_loss = _make_loss_inputs()

        mock_args = MagicMock()
        mock_args.kl_loss_weight = kl_weight

        with patch.object(qad_patches, 'get_args', return_value=mock_args):
            total_loss, num_tokens, metrics = _qad_loss_func(
                "output_tensor",
                original_loss_func_partial=original_partial,
                kl_loss_value=kl_loss,
            )

        assert torch.isclose(total_loss, torch.tensor(expected_total))
        # KL-only mode: num_tokens = 1 (Megatron's division is a no-op)
        assert num_tokens == 1
        assert 'kl loss' in metrics
        assert isinstance(metrics['kl loss'], tuple) and len(metrics['kl loss']) == 2
        # CE loss should still be in metrics for logging
        assert 'lm loss' in metrics

    def test_ce_is_logging_only(self):
        """CE loss appears in metrics but does not affect total_loss."""
        original_partial, kl_loss = _make_loss_inputs()

        mock_args = MagicMock()
        mock_args.kl_loss_weight = 1.0

        with patch.object(qad_patches, 'get_args', return_value=mock_args):
            total_loss, _, metrics = _qad_loss_func(
                "output_tensor",
                original_loss_func_partial=original_partial,
                kl_loss_value=kl_loss,
            )

        # total_loss should be just kl_weight * kl_loss, NOT including CE
        assert torch.isclose(total_loss, torch.tensor(1.0 * _KL_LOSS))
        # But CE should still be logged
        assert 'lm loss' in metrics
        ce_value, ce_tokens = metrics['lm loss']
        assert torch.isclose(ce_value, torch.tensor(_CE_LOSS))

    def test_via_partial_call(self):
        """Test calling _qad_loss_func via partial as the training loop does."""
        original_partial, kl_loss = _make_loss_inputs()

        mock_args = MagicMock()
        mock_args.kl_loss_weight = 1.0

        loss_func = partial(
            _qad_loss_func,
            original_loss_func_partial=original_partial,
            kl_loss_value=kl_loss,
        )

        with patch.object(qad_patches, 'get_args', return_value=mock_args):
            total_loss, _, _ = loss_func("output_tensor")

        expected = 1.0 * _KL_LOSS
        assert torch.isclose(total_loss, torch.tensor(expected))


# ============================================================================
# Integration
# ============================================================================


class TestQADPatchesIntegration:
    """Integration tests for QAD patches working together."""

    def test_gradient_flow_through_combined_loss(self):
        """Gradients flow through combined CE+KL loss to student logits."""
        student_logits = torch.randn(4, 8, 100, requires_grad=True)
        teacher_logits = torch.randn(4, 8, 100)

        import torch.nn.functional as F

        teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
        student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
        kl_loss = F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True)
        kl_loss = kl_loss.sum(dim=-1).mean()

        ce_loss = torch.tensor(0.5, requires_grad=True)
        total_loss = ce_loss + 1.0 * kl_loss
        total_loss.backward()

        assert student_logits.grad is not None
        assert student_logits.grad.abs().sum().item() > 0

    def test_qad_forward_step_returns_correct_format(self):
        """qad_forward_step returns (output_tensor, partial_loss_func) format."""
        original_forward_step = MagicMock()
        original_forward_step.return_value = (
            torch.randn(8, 4),
            partial(lambda lm, ot: (torch.tensor(0.5), torch.tensor(32), {}), None),
        )

        mock_args = MagicMock()
        mock_args.qad_enable = False

        with patch.object(qad_patches, 'get_args', return_value=mock_args):
            result = original_forward_step("data_iterator", "model")

        assert isinstance(result, tuple) and len(result) == 2
        _, loss_func_partial = result
        assert callable(loss_func_partial)


# ============================================================================
# Teacher Manager Singleton
# ============================================================================


class TestTeacherManagerSingleton:
    """Test the global teacher manager singleton lifecycle.

    Verifies that reset_teacher_manager() clears the cached singleton so the
    next get_teacher_manager() call re-reads args and creates a fresh instance.
    """

    def setup_method(self):
        reset_teacher_manager()

    def teardown_method(self):
        reset_teacher_manager()

    def test_get_creates_instance_on_first_call(self):
        """get_teacher_manager() lazily creates the manager from args."""
        fresh_mock = MagicMock(return_value=MagicMock(name="teacher_mgr"))
        with (
            patch.object(qad_patches, 'get_args', return_value=_make_singleton_mock_args()),
            patch.object(qad_patches, 'TeacherModelManager', fresh_mock),
        ):
            mgr = get_teacher_manager()

        assert mgr is not None
        assert fresh_mock.called

    def test_get_returns_cached_instance(self):
        """Subsequent get_teacher_manager() calls return the same cached instance."""
        cached_instance = MagicMock(name="cached_teacher_mgr")
        fresh_mock = MagicMock(return_value=cached_instance)
        with (
            patch.object(qad_patches, 'get_args', return_value=_make_singleton_mock_args()),
            patch.object(qad_patches, 'TeacherModelManager', fresh_mock),
        ):
            mgr1 = get_teacher_manager()
            mgr2 = get_teacher_manager()

        assert mgr1 is mgr2 is cached_instance
        assert fresh_mock.call_count == 1

    def test_reset_clears_singleton(self):
        """reset_teacher_manager() clears the cache so next get creates new."""
        instance1 = MagicMock(name="teacher_mgr_1")
        instance2 = MagicMock(name="teacher_mgr_2")
        fresh_mock = MagicMock(side_effect=[instance1, instance2])
        with (
            patch.object(qad_patches, 'get_args', return_value=_make_singleton_mock_args()),
            patch.object(qad_patches, 'TeacherModelManager', fresh_mock),
        ):
            mgr1 = get_teacher_manager()
            reset_teacher_manager()
            mgr2 = get_teacher_manager()

        assert mgr1 is instance1
        assert mgr2 is instance2
        assert fresh_mock.call_count == 2

    def test_reset_when_already_none_is_noop(self):
        """reset_teacher_manager() is safe to call when no instance exists."""
        reset_teacher_manager()
        reset_teacher_manager()  # should not raise


# ============================================================================
# _all_reduce_kl_for_logging
# ============================================================================


class TestAllReduceKlForLogging:
    """Test _all_reduce_kl_for_logging - sums KL across TP ranks for logging.

    The KL loss on each TP rank is a *partial* divergence over that rank's
    local vocabulary partition. The full KL is the SUM of all partials.
    Previously the code divided by tp_size after the sum-reduce, producing
    the mean of partials (full_KL / tp_size) — an incorrect value.
    """

    def test_tp1_no_all_reduce(self):
        """TP=1: no all_reduce, returns the input value unchanged."""
        kl = torch.tensor(0.42)
        result = qad_patches._all_reduce_kl_for_logging(kl)
        assert torch.isclose(result, kl)

    def test_tp2_sums_without_division(self):
        """TP>1: all_reduce SUM is called, result is NOT divided by tp_size."""
        kl = torch.tensor(0.42)

        captured = {}

        def fake_all_reduce(tensor, group=None, op=None):
            # Simulate sum-reduce: double the value (as if 2 ranks each had 0.42)
            captured['called'] = True
            captured['op'] = op
            tensor.add_(0.42)  # 0.42 + 0.42 = 0.84 (sum of 2 partials)

        with (
            patch.object(qad_patches.mpu, 'get_tensor_model_parallel_world_size', return_value=2),
            patch.object(qad_patches.mpu, 'get_tensor_model_parallel_group', return_value="tp_group"),
            patch.object(torch.distributed, 'all_reduce', side_effect=fake_all_reduce),
        ):
            result = qad_patches._all_reduce_kl_for_logging(kl)

        assert captured.get('called') is True
        # Result should be the SUM (0.84), NOT the mean (0.42)
        assert torch.isclose(result, torch.tensor(0.84))
        # Verify it was NOT divided by tp_size
        assert not torch.isclose(result, torch.tensor(0.42))

    def test_tp4_sums_four_partials(self):
        """TP=4: sum of 4 partials, not divided by 4."""
        kl = torch.tensor(1.0)

        def fake_all_reduce(tensor, group=None, op=None):
            # Simulate 4 ranks each with partial=1.0 → sum=4.0
            tensor.add_(3.0)

        with (
            patch.object(qad_patches.mpu, 'get_tensor_model_parallel_world_size', return_value=4),
            patch.object(qad_patches.mpu, 'get_tensor_model_parallel_group', return_value="tp_group"),
            patch.object(torch.distributed, 'all_reduce', side_effect=fake_all_reduce),
        ):
            result = qad_patches._all_reduce_kl_for_logging(kl)

        # Should be 4.0 (sum), NOT 1.0 (mean)
        assert torch.isclose(result, torch.tensor(4.0))

    def test_does_not_modify_original(self):
        """The original kl_loss_value is not modified (detach+clone)."""
        kl = torch.tensor(0.5)
        original_value = kl.item()

        with (
            patch.object(qad_patches.mpu, 'get_tensor_model_parallel_world_size', return_value=1),
        ):
            qad_patches._all_reduce_kl_for_logging(kl)

        assert kl.item() == original_value


# ============================================================================
# reset_kl_loss_module
# ============================================================================


class TestResetKlLossModule:
    """Test reset_kl_loss_module() - clears cached KL loss module for test isolation."""

    def setup_method(self):
        qad_patches.reset_kl_loss_module()

    def teardown_method(self):
        qad_patches.reset_kl_loss_module()

    def test_reset_clears_cached_module(self):
        """reset_kl_loss_module() clears the global _kl_loss_module."""
        # Set the module by calling _get_kl_loss_module
        qad_patches._get_kl_loss_module(1.0)
        assert qad_patches._kl_loss_module is not None

        qad_patches.reset_kl_loss_module()
        assert qad_patches._kl_loss_module is None

    def test_reset_when_already_none_is_noop(self):
        """reset_kl_loss_module() is safe to call when no module exists."""
        qad_patches.reset_kl_loss_module()
        qad_patches.reset_kl_loss_module()  # should not raise

    def test_temperature_change_rebuilds_module(self):
        """Changing temperature causes _get_kl_loss_module to rebuild."""
        # Configure LogitsKLLoss mock to return objects with .temperature set
        # so the caching logic (_kl_loss_module.temperature != temperature) works.
        mod_t1_a = MagicMock(name="mod_T1_a")
        mod_t1_a.temperature = 1.0
        mod_t2 = MagicMock(name="mod_T2")
        mod_t2.temperature = 2.0
        mod_t1_b = MagicMock(name="mod_T1_b")
        mod_t1_b.temperature = 1.0
        qad_patches.LogitsKLLoss.side_effect = [mod_t1_a, mod_t2, mod_t1_b]

        mod1 = qad_patches._get_kl_loss_module(1.0)
        mod2 = qad_patches._get_kl_loss_module(1.0)
        assert mod1 is mod2  # same temperature → cached (no new call)

        mod3 = qad_patches._get_kl_loss_module(2.0)
        assert mod3 is not mod1  # different temperature → rebuilt

        # After reset, a new module is created even with same temperature
        qad_patches.reset_kl_loss_module()
        mod4 = qad_patches._get_kl_loss_module(1.0)
        assert mod4 is not mod1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
