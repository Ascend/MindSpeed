# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Python-level tests for MindSpeed batch-invariant adapters.

These tests do not validate Ascend kernel accuracy. They replace
``torch.ops.batch_invariant_ops`` with Python reference implementations and
check dispatch, fallback, and autograd behavior in the MindSpeed adapter layer.
Real NPU kernel coverage belongs in ops-batchinvariant examples or NPU smoke
tests.
"""

import sys
import types

import pytest
import torch

from mindspeed.core.batch_invariant import batch_invariant as bik


class FakeBatchInvariantOps:
    @staticmethod
    def npu_mm_batch_invariant(a, b):
        return torch.mm(a, b)

    @staticmethod
    def npu_matmul_batch_invariant(a, b):
        return torch.matmul(a, b)

    @staticmethod
    def npu_reduce_sum_batch_invariant(x, dim, keepdim):
        return bik._ORIGINAL_TORCH_SUM(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def npu_log_softmax_batch_invariant(x, dim):
        return torch.log_softmax(x, dim=dim)


class FakeLibrary:
    instances = []

    def __init__(self, namespace, kind):
        self.namespace = namespace
        self.kind = kind
        self.impls = []
        self.destroyed = False
        FakeLibrary.instances.append(self)

    def impl(self, name, fn, dispatch_key):
        self.impls.append((name, fn, dispatch_key))

    def _destroy(self):
        self.destroyed = True


@pytest.fixture(autouse=True)
def restore_batch_invariant_state():
    yield
    if bik.is_batch_invariant_mode_enabled():
        bik.disable_batch_invariant_mode()


@pytest.fixture
def fake_ops(monkeypatch):
    monkeypatch.setattr(torch.ops, "batch_invariant_ops", FakeBatchInvariantOps(), raising=False)


@pytest.fixture
def fake_npu_device(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: types.SimpleNamespace(type="npu")))


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _assert_same_tensor(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mm_adapter_forward_backward(fake_ops):
    a = torch.randn(4, 3, requires_grad=True)
    b = torch.randn(3, 5, requires_grad=True)
    a_ref = _clone_with_grad(a)
    b_ref = _clone_with_grad(b)

    out = bik.mm_adapter(a, b)
    out_ref = torch.mm(a_ref, b_ref)
    _assert_same_tensor(out, out_ref)

    out.sum().backward()
    out_ref.sum().backward()
    _assert_same_tensor(a.grad, a_ref.grad)
    _assert_same_tensor(b.grad, b_ref.grad)


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((4, 3), (3, 5)),
        ((2, 4, 3), (2, 3, 5)),
        ((2, 4, 3), (3, 5)),
        ((3,), (3,)),
        ((3,), (3, 5)),
        ((4, 3), (3,)),
        ((3,), (2, 3, 5)),
        ((2, 4, 3), (3,)),
    ],
)
def test_matmul_adapter_forward_backward(fake_ops, a_shape, b_shape):
    a = torch.randn(*a_shape, requires_grad=True)
    b = torch.randn(*b_shape, requires_grad=True)
    a_ref = _clone_with_grad(a)
    b_ref = _clone_with_grad(b)

    out = bik.matmul_adapter(a, b)
    out_ref = torch.matmul(a_ref, b_ref)
    _assert_same_tensor(out, out_ref)

    out.sum().backward()
    out_ref.sum().backward()
    _assert_same_tensor(a.grad, a_ref.grad)
    _assert_same_tensor(b.grad, b_ref.grad)


@pytest.mark.parametrize("dim,keepdim", [(-1, True), (-1, False), (1, True), (1, False), ((1,), True)])
def test_reduce_sum_adapter_forward_backward(fake_ops, fake_npu_device, dim, keepdim):
    x = torch.randn(2, 3, 4, requires_grad=True)
    x_ref = _clone_with_grad(x)

    out = bik.reduce_sum_adapter(x, dim=dim, keepdim=keepdim)
    out_ref = torch.sum(x_ref, dim=dim, keepdim=keepdim)
    _assert_same_tensor(out, out_ref)

    out.sum().backward()
    out_ref.sum().backward()
    _assert_same_tensor(x.grad, x_ref.grad)


def test_reduce_sum_adapter_fallback_for_multi_dim_tuple():
    x = torch.randn(2, 3, 4, requires_grad=True)

    out = bik.reduce_sum_adapter(x, dim=(1, 2), keepdim=True)
    out_ref = torch.sum(x, dim=(1, 2), keepdim=True)
    _assert_same_tensor(out, out_ref)


def test_reduce_sum_adapter_fallback_for_integer_dtype(fake_npu_device):
    x = torch.ones(2, 3, dtype=torch.int64)

    out = bik.reduce_sum_adapter(x, dim=1)
    out_ref = torch.sum(x, dim=1)
    _assert_same_tensor(out, out_ref)


def test_log_softmax_adapter_forward_backward(fake_ops):
    x = torch.randn(3, 5, requires_grad=True)
    x_ref = _clone_with_grad(x)

    out = bik.log_softmax_adapter(x, dim=-1)
    out_ref = torch.log_softmax(x_ref, dim=-1)
    _assert_same_tensor(out, out_ref)

    out.sum().backward()
    out_ref.sum().backward()
    _assert_same_tensor(x.grad, x_ref.grad)


def test_internal_log_softmax_adapter_accepts_half_to_float(fake_ops):
    x = torch.randn(3, 5, dtype=torch.float16, requires_grad=True)

    out = bik._log_softmax_adapter(x, dim=-1, half_to_float=True)

    assert out.dtype == torch.float32
    _assert_same_tensor(out, torch.log_softmax(x.float(), dim=-1))


def test_enable_disable_batch_invariant_mode(monkeypatch):
    FakeLibrary.instances.clear()
    monkeypatch.setitem(sys.modules, "batch_invariant_ops", types.ModuleType("batch_invariant_ops"))
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(torch.library, "Library", FakeLibrary)

    original_torch_sum = torch.sum
    original_tensor_sum = torch.Tensor.sum

    bik.enable_batch_invariant_mode()

    assert bik.is_batch_invariant_mode_enabled()
    assert torch.sum is bik.reduce_sum_adapter
    assert torch.Tensor.sum is bik.reduce_sum_adapter
    assert len(FakeLibrary.instances) == 1
    assert FakeLibrary.instances[0].impls == [
        ("aten::mm", bik.mm_adapter, "NPU"),
        ("aten::matmul", bik.matmul_adapter, "NPU"),
        ("aten::sum", bik.reduce_sum_adapter, "NPU"),
        ("aten::_log_softmax", bik._log_softmax_adapter, "NPU"),
        ("aten::log_softmax", bik.log_softmax_adapter, "NPU"),
    ]

    bik.disable_batch_invariant_mode()

    assert not bik.is_batch_invariant_mode_enabled()
    assert FakeLibrary.instances[0].destroyed
    assert torch.sum is original_torch_sum
    assert torch.Tensor.sum is original_tensor_sum
