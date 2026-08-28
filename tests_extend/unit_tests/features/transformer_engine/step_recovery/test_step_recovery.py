#!/usr/bin/env python3
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the HiF8 Step Recovery feature.

These tests are pure-Python and do NOT require NPU / CUDA / distributed
initialization.  They mock the device-side and collective primitives so the
controller logic can be exercised deterministically on CPU.
"""

import random
from typing import NamedTuple, Optional
from unittest import mock

import numpy as np
import pytest
import torch

from mindspeed.core.transformer_engine.step_recovery.controller import RecoverableStepRunner
from mindspeed.core.transformer_engine.step_recovery.diagnostics import StepRecoveryDiagnostics
from mindspeed.core.transformer_engine.step_recovery.recovery import (
    DistributedFailureSync,
    ExternalReplayAdapter,
    FailedAttemptCleanup,
    HiF8ResetAdapter,
    RNGStateManager,
    ReplayableIterator,
    StepRecovery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyOptimizer:
    """Minimal optimizer stub with a found_inf buffer."""

    def __init__(self):
        self.found_inf = torch.zeros(1, device="cpu")
        self._params = [torch.zeros(4, requires_grad=False)]

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        return True


class TrainStepResult(NamedTuple):
    """Named tuple matching the return contract of ``train_step``.

    ``(loss_dict, skipped_iter, should_checkpoint, should_exit,
      exit_code, grad_norm, num_zeros_in_grad, log_max_attention_logit)``
    """

    loss_dict: dict
    skipped_iter: int
    should_checkpoint: bool
    should_exit: bool
    exit_code: int
    grad_norm: Optional[torch.Tensor]
    num_zeros_in_grad: int
    log_max_attention_logit: float


def _make_result(loss=1.0, skipped_iter=0, grad_norm=1.0, should_exit=False):
    """Build a train_step-shaped return NamedTuple."""
    loss_dict = {"lm loss": torch.tensor([loss], dtype=torch.float32)}
    return TrainStepResult(
        loss_dict=loss_dict,
        skipped_iter=skipped_iter,
        should_checkpoint=False,
        should_exit=should_exit,
        exit_code=0,
        grad_norm=torch.tensor([grad_norm], dtype=torch.float32) if grad_norm is not None else None,
        num_zeros_in_grad=0,
        log_max_attention_logit=0.0,
    )


def _make_runner(*, hif8_calls=None):
    """Build a runner with mocked distributed + HiF8 reset tracking."""
    hif8_calls = hif8_calls if hif8_calls is not None else []
    adapter = HiF8ResetAdapter()
    adapter.reset_after_failed_step = lambda: hif8_calls.append(True)
    recovery = StepRecovery(
        cleanup=FailedAttemptCleanup(),
        hif8_adapter=adapter,
    )

    # Single-process failure sync (no torch.distributed).
    sync = DistributedFailureSync()

    return RecoverableStepRunner(
        recovery=recovery,
        diagnostics=StepRecoveryDiagnostics(),
    ), sync


# ---------------------------------------------------------------------------
# Batch replay tests
# ---------------------------------------------------------------------------


class TestBatchReplay:
    def test_single_iterator_replay(self):
        source = iter([{"x": torch.tensor([1.0])},
                       {"x": torch.tensor([2.0])},
                       {"x": torch.tensor([3.0])}])
        mgr = ExternalReplayAdapter(source)
        it = mgr.get_original_iterator()
        first = [next(it), next(it), next(it)]
        mgr.rollback()
        replay_it = mgr.get_retry_iterator()
        replayed = [next(replay_it), next(replay_it), next(replay_it)]
        assert all(torch.equal(a["x"], b["x"]) for a, b in zip(first, replayed))
        mgr.commit()

    def test_replay_does_not_consume_source(self):
        source = iter([{"x": torch.tensor([1.0])},
                       {"x": torch.tensor([2.0])}])
        mgr = ExternalReplayAdapter(source)
        it0 = mgr.get_original_iterator()
        next(it0); next(it0)
        mgr.rollback()
        it1 = mgr.get_retry_iterator()
        next(it1); next(it1)
        with pytest.raises(RuntimeError):
            next(it1)

    def test_list_of_iterators(self):
        src_a = iter([torch.tensor([1.0]), torch.tensor([2.0])])
        src_b = iter([torch.tensor([10.0]), torch.tensor([20.0])])
        mgr = ExternalReplayAdapter([src_a, src_b])
        its = mgr.get_original_iterator()
        assert isinstance(its, list) and len(its) == 2
        a0 = next(its[0]); b0 = next(its[1])
        mgr.rollback()
        its2 = mgr.get_retry_iterator()
        a0r = next(its2[0]); b0r = next(its2[1])
        assert torch.equal(a0, a0r)
        assert torch.equal(b0, b0r)

    def test_inplace_mutation_does_not_corrupt_replay(self):
        source = iter([{"x": torch.zeros(2)}])
        mgr = ExternalReplayAdapter(source)
        it = mgr.get_original_iterator()
        batch = next(it)
        batch["x"].add_(99.0)  # in-place mutation
        mgr.rollback()
        it2 = mgr.get_retry_iterator()
        replayed = next(it2)
        assert torch.equal(replayed["x"], torch.zeros(2))

    def test_list_of_iterators_preserves_none(self):
        src = iter([torch.tensor([1.0])])
        mgr = ExternalReplayAdapter([None, src])

        its = mgr.get_original_iterator()
        assert its[0] is None
        first = next(its[1])

        mgr.rollback()
        retry_its = mgr.get_retry_iterator()
        assert retry_its[0] is None
        replayed = next(retry_its[1])

        assert torch.equal(first, replayed)


# ---------------------------------------------------------------------------
# DistributedFailureSync tests
# ---------------------------------------------------------------------------


class TestDistributedFailureSync:
    def test_single_process_no_failure_returns_false_none(self):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import (
            RetryReason,
        )

        sync = DistributedFailureSync()
        global_failed, global_reason = sync.sync_failure(
            False, RetryReason.NONE,
        )
        assert global_failed is False
        assert global_reason is RetryReason.NONE

    def test_single_process_failure_returns_true_with_reason(self):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import (
            RetryReason,
        )

        sync = DistributedFailureSync()
        global_failed, global_reason = sync.sync_failure(
            True, RetryReason.LOSS_NAN,
        )
        assert global_failed is True
        assert global_reason is RetryReason.LOSS_NAN

    def test_single_process_failure_with_grad_nan_reason(self):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import (
            RetryReason,
        )

        sync = DistributedFailureSync()
        global_failed, global_reason = sync.sync_failure(
            True, RetryReason.GRAD_NAN,
        )
        assert global_failed is True
        assert global_reason is RetryReason.GRAD_NAN

    def test_any_rank_failed_backward_compat(self):
        sync = DistributedFailureSync()
        assert sync.any_rank_failed(False) is False
        assert sync.any_rank_failed(True) is True


# ---------------------------------------------------------------------------
# HiF8 adapter tests
# ---------------------------------------------------------------------------


class TestHiF8ResetAdapter:
    def test_calls_reset_when_available(self):
        adapter = HiF8ResetAdapter()
        called = []

        class _FakeFP8GlobalStateManager:
            @staticmethod
            def reset_fp8_amax_history():
                called.append(True)

        fake_module = type("m", (), {"FP8GlobalStateManager": _FakeFP8GlobalStateManager})

        with mock.patch.dict("sys.modules", {
            "transformer_engine": type("m", (), {}),
            "transformer_engine.pytorch": type("m", (), {}),
            "transformer_engine.pytorch.quantization": fake_module,
        }):
            adapter.reset_after_failed_step()
        assert called == [True]

    def test_missing_te_is_safe(self):
        adapter = HiF8ResetAdapter()

        with mock.patch.dict("sys.modules", {}):
            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name.startswith("transformer_engine"):
                    raise ImportError("simulated absence")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                adapter.reset_after_failed_step()  # must not raise


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------


class TestFailedAttemptCleanup:
    def test_clears_optimizer_found_inf(self):
        opt = _DummyOptimizer()
        opt.found_inf.fill_(1.0)
        FailedAttemptCleanup().cleanup(model=None, optimizer=opt)
        assert float(opt.found_inf) == 0.0

    def test_handles_missing_optimizer(self):
        FailedAttemptCleanup().cleanup(model=None, optimizer=None)

    def test_clears_chained_optimizer_found_inf(self):
        opt = _DummyOptimizer()
        opt.found_inf.fill_(1.0)
        opt.chained_optimizers = [_DummyOptimizer()]
        opt.chained_optimizers[0].found_inf.fill_(1.0)
        FailedAttemptCleanup().cleanup(model=None, optimizer=opt)
        assert float(opt.found_inf) == 0.0
        assert float(opt.chained_optimizers[0].found_inf) == 0.0


# ---------------------------------------------------------------------------
# RNG state tests
# ---------------------------------------------------------------------------


class TestRNGStateManager:
    def test_capture_restore_roundtrip(self):
        mgr = RNGStateManager()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        snap = mgr.capture()

        # Advance RNG.
        random.random(); np.random.rand(); torch.rand(1)

        mgr.restore(snap)
        a1 = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        mgr.restore(snap)
        a2 = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert a1 == a2


# ---------------------------------------------------------------------------
# Feature / patch wiring tests
# ---------------------------------------------------------------------------


class TestFeatureWiring:
    def test_te_recipe_registers_no_hif8_step_recovery_arg(self):
        import argparse

        from mindspeed.features_manager.transformer_engine.te_recipe import (
            TeRecipeFeature,
        )

        parser = argparse.ArgumentParser()
        TeRecipeFeature().register_args(parser)
        args = parser.parse_args(["--no-hif8-step-recovery"])
        assert args.no_hif8_step_recovery is True

    def test_te_recipe_registers_train_step_patch(self):
        from argparse import Namespace

        from mindspeed.features_manager.transformer_engine.te_recipe import (
            TeRecipeFeature,
        )

        feature = TeRecipeFeature()
        mock_pm = mock.Mock()
        feature.register_patches(mock_pm, Namespace())
        registered_targets = [call.args[0] for call in mock_pm.register_patch.call_args_list]
        assert "megatron.training.training.train_step" in registered_targets

    def test_wrapper_passthrough_when_recipe_not_hif8_delayed(self):
        from mindspeed.core.transformer_engine.step_recovery.patch import (
            train_step_recovery_wrapper,
        )

        called = []

        def fake_train_step(*a, **kw):
            called.append(True)
            return _make_result()

        wrapped = train_step_recovery_wrapper(fake_train_step)

        fake_args = type("ns", (), {
            "fp8_recipe": "delayed",
            "no_hif8_step_recovery": False,
        })()
        with mock.patch(
            "mindspeed.core.transformer_engine.step_recovery.patch.get_full_args",
            return_value=fake_args,
        ):
            result = wrapped(None, iter([]), None, None, None, None, None, iteration=0)
        assert len(called) == 1
        assert not result.skipped_iter

    def test_wrapper_passthrough_when_explicitly_disabled(self):
        from mindspeed.core.transformer_engine.step_recovery.patch import (
            train_step_recovery_wrapper,
        )

        called = []

        def fake_train_step(*a, **kw):
            called.append(True)
            return _make_result()

        wrapped = train_step_recovery_wrapper(fake_train_step)

        fake_args = type("ns", (), {
            "fp8_recipe": "hif8_delayed",
            "no_hif8_step_recovery": True,
        })()
        with mock.patch(
            "mindspeed.core.transformer_engine.step_recovery.patch.get_full_args",
            return_value=fake_args,
        ):
            result = wrapped(None, iter([]), None, None, None, None, None, iteration=0)
        assert len(called) == 1
        assert not result.skipped_iter

    def test_wrapper_active_when_hif8_delayed_recipe(self):
        from mindspeed.core.transformer_engine.step_recovery.patch import (
            train_step_recovery_wrapper,
        )

        def fake_train_step(*a, **kw):
            return _make_result()

        wrapped = train_step_recovery_wrapper(fake_train_step)

        fake_args = type("ns", (), {
            "fp8_recipe": "hif8_delayed",
            "no_hif8_step_recovery": False,
        })()
        with mock.patch(
            "mindspeed.core.transformer_engine.step_recovery.patch.get_full_args",
            return_value=fake_args,
        ), mock.patch(
            "mindspeed.core.transformer_engine.step_recovery.patch.RecoverableStepRunner",
        ) as mock_runner_cls:
            mock_runner = mock.MagicMock()
            mock_runner.run_train_step.return_value = _make_result()
            mock_runner_cls.return_value = mock_runner
            wrapped(None, iter([]), None, None, None, None, None, iteration=0)
        mock_runner.run_train_step.assert_called_once()

    def test_hif8_delayed_requires_hif8_format(self):
        from argparse import Namespace

        from mindspeed.features_manager.transformer_engine.te_recipe import (
            TeRecipeFeature,
        )

        args = Namespace(
            fp8='e4m3',
            fp8_recipe='hif8_delayed',
            use_gmm_fp8=False,
            fp8_reuse_quantized_weight=False,
        )

        with pytest.raises(ValueError, match="hif8_delayed recipe requires"):
            TeRecipeFeature().validate_args(args)


# ---------------------------------------------------------------------------
# Pre-commit guard tests
# ---------------------------------------------------------------------------


class FakeConfig:
    clip_grad = 1.0
    log_num_zeros_in_grad = True


class FakeOptimizer:
    """Minimal optimizer with prepare_grads / clip_grad_norm / step_with_ready_grads."""

    def __init__(self):
        self.config = FakeConfig()
        self.prepare_grads_return = False
        self.grad_norm_return = torch.tensor(1.0)

        self.prepare_grads_called = 0
        self.clip_grad_norm_called = 0
        self.step_with_ready_grads_called = 0

    def prepare_grads(self):
        self.prepare_grads_called += 1
        return self.prepare_grads_return

    def clip_grad_norm(self, clip_grad):
        self.clip_grad_norm_called += 1
        return self.grad_norm_return

    def count_zeros(self):
        return 0

    def step_with_ready_grads(self):
        self.step_with_ready_grads_called += 1
        return True

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        found_inf = self.prepare_grads()
        if found_inf:
            return False, None, None

        grad_norm = self.clip_grad_norm(self.config.clip_grad)
        num_zeros = self.count_zeros()
        success = self.step_with_ready_grads()

        return success, grad_norm, num_zeros


class FakeFailureSync:
    def sync_failure(self, local_failed, local_reason):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import RetryReason

        if local_failed:
            return True, local_reason
        return False, RetryReason.NONE

    def any_rank_failed(self, local_failed):
        return bool(local_failed)


class TestPreCommitGuard:
    def test_loss_nan_blocks_commit_at_barrier(self):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import (
            RetryableStepError,
            RetryReason,
        )
        from mindspeed.core.transformer_engine.step_recovery.pre_commit_guard import (
            StepHealth,
            make_guarded_forward_backward_func,
            patch_optimizer_pre_commit_methods,
        )

        opt = FakeOptimizer()
        health = StepHealth()

        guarded_fb = make_guarded_forward_backward_func(
            lambda *a, **k: {"lm loss": torch.tensor(float("nan"))},
            health,
        )

        guarded_fb()

        with patch_optimizer_pre_commit_methods(opt, FakeFailureSync(), health):
            with pytest.raises(RetryableStepError):
                opt.step()

        assert opt.step_with_ready_grads_called == 0

    def test_prepare_grads_found_inf_raises_before_commit(self):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import (
            RetryableStepError,
            RetryReason,
        )
        from mindspeed.core.transformer_engine.step_recovery.pre_commit_guard import (
            StepHealth,
            patch_optimizer_pre_commit_methods,
        )

        opt = FakeOptimizer()
        opt.prepare_grads_return = True

        health = StepHealth()

        with patch_optimizer_pre_commit_methods(opt, FakeFailureSync(), health):
            with pytest.raises(RetryableStepError) as exc:
                opt.step()

        assert exc.value.reason == RetryReason.OPTIMIZER_OVERFLOW
        assert opt.step_with_ready_grads_called == 0

    def test_grad_norm_nan_blocks_commit(self):
        from mindspeed.core.transformer_engine.step_recovery.exceptions import (
            RetryableStepError,
            RetryReason,
        )
        from mindspeed.core.transformer_engine.step_recovery.pre_commit_guard import (
            StepHealth,
            patch_optimizer_pre_commit_methods,
        )

        opt = FakeOptimizer()
        opt.prepare_grads_return = False
        opt.grad_norm_return = torch.tensor(float("nan"))

        health = StepHealth()

        with patch_optimizer_pre_commit_methods(opt, FakeFailureSync(), health):
            with pytest.raises(RetryableStepError) as exc:
                opt.step()

        assert exc.value.reason == RetryReason.GRAD_NAN
        assert opt.step_with_ready_grads_called == 0

    def test_normal_step_commits_once(self):
        from mindspeed.core.transformer_engine.step_recovery.pre_commit_guard import (
            StepHealth,
            patch_optimizer_pre_commit_methods,
        )

        opt = FakeOptimizer()
        opt.prepare_grads_return = False
        opt.grad_norm_return = torch.tensor(1.0)

        health = StepHealth()

        with patch_optimizer_pre_commit_methods(opt, FakeFailureSync(), health):
            result = opt.step()

        assert result[0] is True
        assert opt.prepare_grads_called == 1
        assert opt.clip_grad_norm_called == 1
        assert opt.step_with_ready_grads_called == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])