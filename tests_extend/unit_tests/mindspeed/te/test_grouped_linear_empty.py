# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import mindspeed.megatron_adaptor  # noqa: F401
from megatron.core.tensor_parallel.mappings import _AllToAll
from mindspeed.te.pytorch.module.grouped_linear import (
    MindSpeedTEGroupedLinearMXFP8GMM,
    MindSpeedTEGroupedLinearMXFP832x32GMM,
)


@pytest.fixture(name="fp8_gmm", params=[MindSpeedTEGroupedLinearMXFP8GMM, MindSpeedTEGroupedLinearMXFP832x32GMM])
def _fp8_gmm(request, monkeypatch):
    """Use CPU matmuls for the kernels, retaining the real TE autograd functions."""
    cls = request.param
    calls = dict(forward=0, dx=0, dw=0)

    def forward(ctx, x, weight_factory, group_list, **kwargs):
        calls["forward"] += 1
        ctx.reference_weights = weight_factory()
        ctx.reference_splits = torch.diff(group_list, prepend=group_list.new_zeros(1)).tolist()
        chunks = x.split(ctx.reference_splits)
        return [torch.cat([chunk @ weight for chunk, weight in zip(chunks, ctx.reference_weights)])]

    def dx(ctx, grad, weight, group_list):
        assert grad.shape[0] > 0, "empty backward must not call the MXFP8 dgrad kernel"
        calls["dx"] += 1
        chunks = grad.split(ctx.reference_splits)
        return [torch.cat([chunk @ w.T for chunk, w in zip(chunks, ctx.reference_weights)])]

    def dw(ctx, x, grad, group_list):
        assert x.shape[0] > 0, "empty backward must not call the zero-K MXFP8 wgrad kernel"
        calls["dw"] += 1
        return [torch.stack([a.T @ b for a, b in zip(x.split(ctx.reference_splits), grad.split(ctx.reference_splits))])]

    monkeypatch.setattr(cls, "op_forward", staticmethod(forward))
    monkeypatch.setattr(cls, "op_dx", staticmethod(dx))
    monkeypatch.setattr(cls, "op_dw", staticmethod(dw))
    return cls, calls


def _linear(cls, x, splits, weights):
    return cls.apply(x, splits, weights, *[weight.T.contiguous() for weight in weights])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_experts", [1, 2, 4])
@pytest.mark.parametrize("accumulate", [False, True])
def test_empty_backward_returns_all_gradients(fp8_gmm, dtype, num_experts, accumulate):
    cls, calls = fp8_gmm
    x = torch.empty(0, 4, dtype=dtype, requires_grad=True)
    weights = [torch.randn(6, 4, dtype=dtype, requires_grad=True) for _ in range(num_experts)]
    splits = torch.zeros(num_experts, dtype=torch.int64)
    if accumulate:
        for weight in weights:
            weight.grad = torch.ones_like(weight)

    output = _linear(cls, x, splits, weights)
    assert output.shape == (0, 6)
    output.sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.empty_like(x))
    for weight in weights:
        assert weight.grad is not None
        expected = torch.ones_like(weight) if accumulate else torch.zeros_like(weight)
        torch.testing.assert_close(weight.grad, expected)
    assert calls == dict(forward=1, dx=0, dw=0)


@pytest.mark.parametrize("counts", [(2, 1), (3, 0), (0, 3)])
def test_nonempty_backward_preserves_kernel_path(fp8_gmm, counts):
    cls, calls = fp8_gmm
    x = torch.randn(sum(counts), 4, requires_grad=True)
    weights = [torch.randn(6, 4, requires_grad=True) for _ in counts]
    grad_output = torch.randn(sum(counts), 6)
    reference = torch.cat([chunk @ weight.T for chunk, weight in zip(x.split(counts), weights)])
    expected = torch.autograd.grad(reference, (x, *weights), grad_output)

    output = _linear(cls, x, torch.tensor(counts), weights)
    torch.testing.assert_close(output, reference)
    output.backward(grad_output)

    for variable, grad in zip((x, *weights), expected):
        torch.testing.assert_close(variable.grad, grad)
    assert calls == dict(forward=1, dx=1, dw=1)


@pytest.mark.parametrize("recompute", [False, True])
def test_empty_experts_preserve_probability_and_alltoall_backward(fp8_gmm, monkeypatch, recompute):
    cls, calls = fp8_gmm
    exchanges = []

    def empty_alltoall(output, input_, output_split_sizes, input_split_sizes, group):
        assert output_split_sizes == input_split_sizes == [0, 0]
        assert output.numel() == input_.numel() == 0
        exchanges.append(tuple(input_.shape))

    # Exercise the real AllToAll autograd boundary without an NPU process group.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", empty_alltoall)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    x = torch.empty(0, 4, requires_grad=True)
    probs = torch.empty(0, requires_grad=True)
    fc1 = [torch.randn(12, 4, requires_grad=True) for _ in range(2)]
    fc2 = [torch.randn(4, 6, requires_grad=True) for _ in range(2)]
    counts = torch.zeros(2, dtype=torch.int64)
    group = object()
    dispatched_x = _AllToAll.apply(group, x, [0, 0], [0, 0])
    dispatched_probs = _AllToAll.apply(group, probs, [0, 0], [0, 0])

    def experts(hidden, probabilities):
        gate, value = _linear(cls, hidden, counts, fc1).chunk(2, dim=-1)
        intermediate = F.silu(gate) * value * probabilities.unsqueeze(-1)
        return _linear(cls, intermediate, counts, fc2)

    if recompute:
        output = checkpoint(experts, dispatched_x, dispatched_probs, use_reentrant=True)
    else:
        output = experts(dispatched_x, dispatched_probs)
    output = _AllToAll.apply(group, output, [0, 0], [0, 0])
    assert len(exchanges) == 3
    exchanges.clear()
    output.sum().backward()

    assert sorted(exchanges) == [(0,), (0, 4), (0, 4)]
    assert x.grad is not None and probs.grad is not None
    torch.testing.assert_close(x.grad, torch.empty_like(x))
    torch.testing.assert_close(probs.grad, torch.empty_like(probs))
    for weight in fc1 + fc2:
        assert weight.grad is not None
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
    assert calls == dict(forward=4 if recompute else 2, dx=0, dw=0)
