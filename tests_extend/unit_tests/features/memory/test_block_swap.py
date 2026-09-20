# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""Unit tests for the block-swap API.

Covers:
1. block semantics: materialize on enter, release on exit, value correctness,
   backward *outside* the block (saved-tensor hooks re-materialization);
2. nesting, multiple tensors, copy_back / version-counter auto refresh,
   guards (CPU tensors, views, empty args, exception safety, release-in-block);
3. Megatron training-loop integration: block-wrapped custom matmul on a model
   weight is updated correctly by the Megatron optimizer (mirrors refreshed,
   no stale re-materialization), numerically equivalent to the baseline run.
"""

import warnings
from functools import partial

import pytest
import torch

from mindspeed import megatron_adaptor  # noqa: F401
from mindspeed.megatron_adaptor import repatch
from mindspeed.core.memory.block_swap import (
    block_swap,
    block_swap_all_in,
    block_swap_all_out,
    get_swap_manager,
)
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.timers import DummyTimer
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model

from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import set_random_seed, initialize_model_parallel

_DEVICE = 'npu' if torch.npu.is_available() else ('cuda' if torch.cuda.is_available() else None)

pytestmark = pytest.mark.skipif(_DEVICE is None, reason='block_swap tests need an accelerator device')


def _make_weight(rows, cols, seed=0, requires_grad=True):
    torch.manual_seed(seed)
    w = torch.randn(rows, cols, device=_DEVICE, dtype=torch.float32)
    w.requires_grad_(requires_grad)
    return w


class TestBlockSwap:
    def test_inference_block(self):
        w = _make_weight(64, 32, seed=1, requires_grad=False)
        reference = w.detach().clone()
        with torch.no_grad(), block_swap(w):
            # materialized inside the block
            assert w.storage().size() != 0
            assert torch.equal(w, reference)
        # released after the block; mirror holds the values
        assert w.storage().size() == 0
        assert torch.equal(w.block_swap_state.cpu, reference.cpu())
        # re-entering re-materializes the same values
        with torch.no_grad(), block_swap(w):
            assert torch.equal(w, reference)
        assert w.storage().size() == 0

    def test_raw_matmul_backward_outside_block(self):
        # THE key scenario: plain torch.matmul inside the block, backward later
        w = _make_weight(64, 32, seed=2)
        x = torch.randn(8, 64, device=_DEVICE, requires_grad=True)
        w_ref = w.detach().clone().requires_grad_(True)
        x_ref = x.detach().clone().requires_grad_(True)

        with block_swap(w):
            assert w.storage().size() != 0
            out = torch.matmul(x, w)
        # the block released the weight storage
        assert w.storage().size() == 0

        # backward OUTSIDE the block: the saved-tensor unpack hook must
        # re-materialize w before matmul's backward reads it
        out.sum().backward()
        ref = torch.matmul(x_ref, w_ref)
        ref.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(w.grad, w_ref.grad, atol=1e-5, rtol=1e-5)
        # the unpack materialized w; it stays materialized until next release
        assert w.storage().size() != 0

    def test_no_grad_saves_nothing(self):
        # under no_grad nothing is saved: exit simply releases, no backward
        w = _make_weight(64, 32, seed=3)
        with torch.no_grad(), block_swap(w):
            out = torch.matmul(torch.randn(8, 64, device=_DEVICE), w)
        assert w.storage().size() == 0
        assert out is not None

    def test_nested_blocks(self):
        w = _make_weight(64, 32, seed=4, requires_grad=False)
        with block_swap(w):
            with block_swap(w):
                assert w.storage().size() != 0
            # inner exit must NOT release (outer scope still active)
            assert w.storage().size() != 0
        assert w.storage().size() == 0

    def test_multiple_tensors_and_copy_back(self):
        w1 = _make_weight(64, 32, seed=5, requires_grad=False)
        w2 = _make_weight(32, 16, seed=6, requires_grad=False)
        original = w1.detach().clone()
        with block_swap(w1, w2, copy_back=True):
            w1.data.add_(1.0)  # in-place modification inside the block
        # copy_back=True refreshed the mirror (compare against the pre-block
        # reference: the device tensor itself is released now)
        assert torch.allclose(w1.block_swap_state.cpu, original.cpu() + 1.0)
        assert w1.storage().size() == 0 and w2.storage().size() == 0

    def test_lazy_state_reused_and_manager_registered(self):
        w = _make_weight(16, 8, seed=7, requires_grad=False)
        assert not hasattr(w, 'block_swap_state')
        with block_swap(w):
            state = w.block_swap_state
            assert any(s is state for s in get_swap_manager().states)
        # second block reuses the same state (no new mirror)
        with block_swap(w):
            assert w.block_swap_state is state

    def test_view_tensor_warns(self):
        base = _make_weight(64, 32, seed=8, requires_grad=False)
        view = base[:32]  # slice: storage_offset 0 but smaller span
        with pytest.warns(UserWarning):
            with block_swap(view):
                pass

    def test_cpu_tensor_rejected(self):
        w = torch.randn(4, 4)
        with pytest.raises(ValueError):
            block_swap(w)

    def test_empty_args_rejected(self):
        with pytest.raises(ValueError):
            block_swap()

    def test_non_tensor_rejected(self):
        with pytest.raises(TypeError):
            block_swap([1, 2, 3])

    def test_exception_inside_block_releases(self):
        w = _make_weight(64, 32, seed=9, requires_grad=False)
        with pytest.raises(RuntimeError):
            with block_swap(w):
                raise RuntimeError('boom')
        assert w.storage().size() == 0

    def test_manual_bulk_control(self):
        w = _make_weight(64, 32, seed=10, requires_grad=False)
        with block_swap(w):
            pass  # adopt into the manager
        block_swap_all_in()
        assert w.storage().size() != 0
        block_swap_all_out(copy_to_host=True)
        assert w.storage().size() == 0

    def test_bulk_release_inside_block_raises(self):
        # an update-window release must refuse tensors with an open scope
        w = _make_weight(16, 8, seed=12, requires_grad=False)
        with block_swap(w):
            with pytest.raises(RuntimeError, match='inside an open block_swap'):
                block_swap_all_out(copy_to_host=True)
        assert w.storage().size() == 0


class TestBlockSwapTraining:
    """In-place semantics of a single adopted tensor (no framework)."""

    def test_auto_copy_back_on_version_change(self):
        w = torch.randn(16, 8, device=_DEVICE)
        original = w.detach().clone()
        with pytest.warns(UserWarning, match='modified in place'):
            with block_swap(w):  # copy_back=False
                w.add_(1.0)  # version-bumping in-place op
        # mirror refreshed automatically
        assert torch.allclose(w.block_swap_state.cpu, original.cpu() + 1.0)
        assert w.storage().size() == 0

    def test_dotdata_write_stays_undetectable_no_false_warning(self):
        # .data.copy_ does not bump the version counter: no auto-refresh and
        # no warning (documented blind spot; use copy_back=True for this)
        w = torch.randn(16, 8, device=_DEVICE)
        original = w.detach().clone()
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning fails the test
            with block_swap(w):
                w.data.fill_(2.0)
        # mirror is stale: it still holds the adoption-time values, not 2.0
        assert torch.equal(w.block_swap_state.cpu, original.cpu())

    def test_adopted_state_survives_manager_reset(self):
        # reset only drops the registry entries; the state stays attached to
        # the tensor and the next block re-adopts it into the manager
        w = _make_weight(16, 8, seed=11, requires_grad=False)
        with block_swap(w):
            pass
        get_swap_manager().reset()
        assert len(get_swap_manager().states) == 0
        with block_swap(w):
            assert any(s is w.block_swap_state for s in get_swap_manager().states)

    def test_block_inside_custom_autograd_function(self):
        # block written INSIDE a custom autograd.Function's forward: the
        # forward pass itself is covered, but saved-tensor hooks are captured
        # at apply() time, NOT inside forward - ctx.save_for_backward is not
        # intercepted (pack/unpack counters stay 0), so backward must
        # re-materialize the weight manually.
        class _CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w):  # pylint: disable=arguments-differ
                with block_swap(w):
                    ctx.save_for_backward(x, w)
                    out = torch.matmul(x, w)
                assert w.storage().size() == 0  # released before forward returns
                return out

            @staticmethod
            def backward(ctx, grad_out):  # pylint: disable=arguments-differ
                x, w = ctx.saved_tensors  # NOT intercepted: still released
                assert w.storage().size() == 0
                w.block_swap_state.swap_in()  # manual re-materialization
                return grad_out.matmul(w.t()), x.t().matmul(grad_out)

        w = _make_weight(64, 32, seed=13)
        x = torch.randn(8, 64, device=_DEVICE, requires_grad=True)
        w_ref = w.detach().clone().requires_grad_(True)
        x_ref = x.detach().clone().requires_grad_(True)

        out = _CustomOp.apply(x, w)
        assert w.storage().size() == 0  # released after apply returns
        out.sum().backward()

        ref = torch.matmul(x_ref, w_ref)
        ref.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(w.grad, w_ref.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Megatron training-loop integration
# ---------------------------------------------------------------------------


class Timers:
    """Minimal stand-in for megatron's Timers (avoids its log args)."""

    def __init__(self):
        self._dummy_timer = DummyTimer()

    def __call__(self, *args, **kwargs):
        return self._dummy_timer


def initialize_gpt_model(pre_process=True, post_process=True, seed=0):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        use_cpu_initialization=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
    )
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=16,
        pre_process=pre_process,
        post_process=post_process,
    )
    model.bfloat16()
    with torch.no_grad():
        for param in model.parameters():
            param.random_()
    return model


def build_model_and_optimizer(seed):
    model = get_model(partial(initialize_gpt_model, seed=seed))
    set_random_seed(seed)
    config = OptimizerConfig(
        lr=1e-4,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
    )
    config.timers = Timers()
    optimizer = get_megatron_optimizer(config, model)
    optimizer.reload_model_params()
    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        unwrapped = unwrapped[0]
    return unwrapped, optimizer


class TestBlockSwapMegatronTraining(DistributedTest):
    """block_swap-wrapped custom matmul on a model weight inside the standard
    Megatron training loop: updates, mirror refreshes and gradients must match
    the baseline (un-wrapped) run.
    """

    world_size = 1

    def test_framework_training_two_steps(self):
        args = parse_args(None, True)
        args.npu_deterministic = False
        args.bf16 = True
        args.accumulate_allreduce_grads_in_fp32 = True
        args.use_distributed_optimizer = False
        args.ddp_bucket_size = None
        args.num_query_groups = None
        args.data_parallel_random_init = False
        args.virtual_pipeline_model_parallel_size = None
        set_args(args)
        repatch(vars(args))
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        def make_input(weight):
            torch.manual_seed(11)
            return torch.randn(4, weight.shape[1], device='cuda', dtype=weight.dtype, requires_grad=True)

        # --- feature run: block-wrapped custom matmul, 2 fwd/bwd/step cycles
        model, optimizer = build_model_and_optimizer(seed=7)
        weight = model.decoder.layers[0].mlp.linear_fc1.weight
        x = make_input(weight)

        for _ in range(2):
            with block_swap(weight):
                out = torch.matmul(x, weight.t())  # custom op, no linear layer
            loss = out.float().pow(2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # the Megatron step wrapper refreshed the mirror and released;
            # re-materialize through the public API to read the device value
            assert weight.storage().size() == 0
            with block_swap(weight):
                device_value = weight.detach().clone().cpu()
            assert torch.equal(weight.block_swap_state.cpu, device_value)
        with block_swap(weight):
            feature_weight = weight.detach().clone().cpu()
        feature_input_grad = x.grad.detach().clone().cpu()

        # --- baseline run: identical flow without the block
        model_base, optimizer_base = build_model_and_optimizer(seed=7)
        weight_base = model_base.decoder.layers[0].mlp.linear_fc1.weight
        x_base = make_input(weight_base)

        for _ in range(2):
            out_base = torch.matmul(x_base, weight_base.t())
            loss_base = out_base.float().pow(2).sum()
            loss_base.backward()
            optimizer_base.step()
            optimizer_base.zero_grad()

        assert torch.allclose(feature_weight, weight_base.detach().cpu(), rtol=1e-2, atol=1e-2), (
            'weights diverged: stale mirror would show here from step 2 on'
        )
        assert torch.allclose(feature_input_grad, x_base.grad.detach().cpu(), rtol=1e-2, atol=1e-2)
