# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import os
import subprocess
import sys

import pytest
import torch

from mindspeed.ops.ffn import npu_ffn
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention, npu_fusion_attention_grad
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")


@pytest.fixture(autouse=True)
def forbid_mindspeed_jit(monkeypatch):
    from mindspeed.op_builder.builder import MindSpeedOpBuilder

    def unexpected_build(*args, **kwargs):
        raise AssertionError("Migrated operators must execute without a MindSpeed JIT extension.")

    monkeypatch.setattr(MindSpeedOpBuilder, "load", unexpected_build)


@pytest.fixture(scope="module")
def isolated_ffn_graph_suite(request):
    # MegatronAdaptor can replace torch.compile process-wide during UT collection.
    if os.environ.get("MINDSPEED_FFN_GRAPH_TEST_CHILD") == "1":
        return False
    nodeids = [
        item.nodeid
        for item in request.session.items
        if item.path == request.path and item.originalname == "test_ffn_ge_graph_tensor_metadata"
    ]
    assert nodeids
    env = os.environ.copy()
    env["MINDSPEED_FFN_GRAPH_TEST_CHILD"] = "1"
    command = [sys.executable, "-m", "pytest", "--run-all", "--import-mode=importlib", "-q", *nodeids]
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=900, check=False, cwd=request.config.rootpath
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def _ffn_inputs(dtype, grouped):
    torch.manual_seed(42)
    x = torch.empty(512, 256, dtype=dtype).uniform_(-1, 1)
    prefix = (8,) if grouped else ()
    w1 = torch.empty(*prefix, 256, 1024, dtype=dtype).uniform_(-0.1, 0.1)
    w2 = torch.empty(*prefix, 1024, 256, dtype=dtype).uniform_(-0.1, 0.1)
    bias_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    b1 = torch.empty(*prefix, 1024, dtype=bias_dtype).uniform_(-0.1, 0.1)
    b2 = torch.empty(*prefix, 256, dtype=bias_dtype).uniform_(-0.1, 0.1)
    return x, w1, w2, b1, b2


def _ffn_reference(inputs, counts=None):
    x, w1, w2, b1, b2 = inputs

    def dense(a, b, c, d, e):
        hidden = torch.relu(a.float() @ b.float() + d.float()).to(a.dtype)
        return (hidden.float() @ c.float() + e.float()).to(a.dtype)

    if counts is None:
        return dense(x, w1, w2, b1, b2)
    outputs = []
    offset = 0
    for i, count in enumerate(counts):
        outputs.append(dense(x[offset : offset + count], w1[i], w2[i], b1[i], b2[i]))
        offset += count
    return torch.cat(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_mode", [None, "expert_tokens", "expert_tokens_index"])
@pytest.mark.parametrize("metadata_device", ["cpu", "npu"])
def test_ffn_tensor_metadata(dtype, token_mode, metadata_device):
    inputs = _ffn_inputs(dtype, token_mode is not None)
    x, w1, w2, b1, b2 = [t.npu() for t in inputs]
    counts = [64, 0, 128, 64, 64, 64, 64, 64] if token_mode else None
    kwargs = {}
    if counts is not None:
        metadata = torch.tensor(counts, dtype=torch.int64, device=metadata_device)
        kwargs[token_mode] = metadata.cumsum(0) if token_mode == "expert_tokens_index" else metadata
    actual = npu_ffn(x, w1, w2, "relu", bias1=b1, bias2=b2, inner_precise=0, **kwargs)
    torch.testing.assert_close(actual.cpu(), _ffn_reference(inputs, counts), rtol=0.005, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_mask", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("token_mode", ["expert_tokens", "expert_tokens_index"])
def test_ffn_reads_updated_device_counts(dtype, bias_mask, token_mode):
    inputs = list(_ffn_inputs(dtype, True))
    if not bias_mask[0]:
        inputs[3].zero_()
    if not bias_mask[1]:
        inputs[4].zero_()
    x, w1, w2, b1, b2 = [value.npu() for value in inputs]
    metadata = torch.empty(8, dtype=torch.int64, device="npu")
    pointer = metadata.data_ptr()
    for counts in ([64] * 8, [64, 0, 128, 64, 64, 64, 64, 64], [64] * 8):
        values = torch.tensor(counts, dtype=torch.int64, device="npu")
        metadata.copy_(values.cumsum(0) if token_mode == "expert_tokens_index" else values)
        actual = npu_ffn(
            x,
            w1,
            w2,
            "relu",
            bias1=b1 if bias_mask[0] else None,
            bias2=b2 if bias_mask[1] else None,
            inner_precise=0,
            **{token_mode: metadata},
        )
        torch.testing.assert_close(actual.cpu(), _ffn_reference(inputs, counts), rtol=0.005, atol=0.005)
        assert metadata.data_ptr() == pointer


def test_ffn_rejects_conflicting_metadata():
    x, w1, w2, b1, b2 = [t.npu() for t in _ffn_inputs(torch.float16, True)]
    counts = torch.tensor([64] * 8, device="npu")
    with pytest.raises(ValueError, match="mutually exclusive"):
        npu_ffn(x, w1, w2, "relu", expert_tokens=counts, expert_tokens_index=counts.cumsum(0))


@pytest.mark.slow
@pytest.mark.parametrize("token_mode", [None, "expert_tokens", "expert_tokens_index"])
def test_ffn_ge_graph_tensor_metadata(token_mode, isolated_ffn_graph_suite):
    if isolated_ffn_graph_suite:
        return
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    inputs = _ffn_inputs(torch.float16, token_mode is not None)
    x, w1, w2, b1, b2 = [t.npu() for t in inputs]

    def forward(x, w1, w2, b1, b2, metadata):
        kwargs = {token_mode: metadata} if token_mode else {}
        return npu_ffn(x, w1, w2, "relu", bias1=b1, bias2=b2, inner_precise=0, **kwargs)

    torch._dynamo.reset()
    compiled_graphs = []
    npu_backend = torchair.get_npu_backend(compiler_config=CompilerConfig())

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return npu_backend(graph, example_inputs)

    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
    metadata = torch.empty(8, dtype=torch.int64, device="npu") if token_mode else None
    pointer = metadata.data_ptr() if metadata is not None else None
    for counts in ([64] * 8, [64, 0, 128, 64, 64, 64, 64, 64], [64] * 8):
        if metadata is not None:
            values = torch.tensor(counts, dtype=torch.int64, device="npu")
            metadata.copy_(values.cumsum(0) if token_mode == "expert_tokens_index" else values)
            assert metadata.data_ptr() == pointer
        actual = compiled(x, w1, w2, b1, b2, metadata)
        expected = _ffn_reference(inputs, counts if token_mode else None)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.005, atol=0.005)
    assert len(compiled_graphs) == 1


def _attention_layout(tensor, layout):
    if layout == "BNSD":
        return tensor
    tensor = tensor.transpose(1, 2).contiguous()
    if layout == "BSH":
        return tensor.flatten(2)
    if layout == "SBH":
        return tensor.flatten(2).transpose(0, 1).contiguous()
    return tensor.squeeze(0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BSH", "SBH", "BNSD", "TND"])
def test_attention_forward_and_backward(dtype, layout):
    torch.manual_seed(42)
    inputs = [torch.randn(1, 32, 128, 128, dtype=dtype) for _ in range(3)]
    reference_inputs = [t.float().requires_grad_() for t in inputs]
    q, k, v = reference_inputs
    expected = torch.softmax((q @ k.transpose(-1, -2)) * 0.08838, dim=-1) @ v
    expected.sum().backward()
    q, k, v = [_attention_layout(t, layout).npu().requires_grad_() for t in inputs]
    kwargs = dict(actual_seq_qlen=[128], actual_seq_kvlen=[128]) if layout == "TND" else {}
    outputs = npu_fusion_attention(q, k, v, 32, layout, scale=0.08838, **kwargs)
    assert len(outputs) == 7
    torch.testing.assert_close(
        outputs[0].cpu(), _attention_layout(expected.detach().to(dtype), layout), rtol=0.02, atol=0.02
    )
    outputs[0].sum().backward()
    for actual, reference in zip((q, k, v), reference_inputs):
        torch.testing.assert_close(
            actual.grad.cpu(), _attention_layout(reference.grad.to(dtype), layout), rtol=0.03, atol=0.03
        )


@pytest.mark.parametrize("keep_prob", [1.0, 0.8])
@pytest.mark.parametrize("pse_type", [0, 1])
def test_attention_explicit_backward_contract(keep_prob, pse_type):
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 32, 128, 128, device="npu", dtype=torch.float16, requires_grad=True) for _ in range(3)]
    pse = torch.randn(1, 32, 128, 128, device="npu", dtype=torch.float16)
    outputs = npu_fusion_attention(q, k, v, 32, "BNSD", pse=pse, pse_type=pse_type, scale=0.08838, keep_prob=keep_prob)
    attention, maximum, total, softmax, seed, offset, numels = outputs
    dy = torch.randn_like(attention)
    actual = npu_fusion_attention_grad(
        q,
        k,
        v,
        dy,
        32,
        "BNSD",
        pse=pse,
        pse_type=pse_type,
        softmax_max=maximum,
        softmax_sum=total,
        softmax_in=softmax,
        attention_in=attention,
        scale=0.08838,
        keep_prob=keep_prob,
        seed=seed,
        offset=offset,
        numels=numels,
    )
    assert len(actual) == 4
    attention.backward(dy)
    # CANN reserves dpse but does not define its contents. Only dq/dk/dv are valid.
    for value, input_tensor in zip(actual[:3], (q, k, v)):
        torch.testing.assert_close(value, input_tensor.grad, rtol=0.001, atol=0.001)


@pytest.mark.parametrize("pse_type", [0, 1])
def test_attention_rejects_unsupported_pse_gradient(pse_type):
    q, k, v = [torch.randn(1, 32, 128, 128, device="npu", dtype=torch.float16) for _ in range(3)]
    pse = torch.randn_like(q, requires_grad=True)
    with pytest.raises(ValueError, match="pse gradients are not supported"):
        npu_fusion_attention(q, k, v, 32, "BNSD", pse=pse, pse_type=pse_type, scale=0.08838)
    with torch.no_grad():
        outputs = npu_fusion_attention(q, k, v, 32, "BNSD", pse=pse, pse_type=pse_type, scale=0.08838)
    assert outputs[0].shape == q.shape


@pytest.mark.parametrize("param_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_adamw_inplace_and_step(param_dtype, grad_dtype, amsgrad, maximize):
    cpu_param = torch.full((10, 10), 0.5)
    cpu_param.requires_grad_()
    optimizer = torch.optim.AdamW(
        [cpu_param], lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8, amsgrad=amsgrad, foreach=False
    )
    param = cpu_param.detach().to(param_dtype).npu()
    m, v = torch.zeros_like(param), torch.zeros_like(param)
    maximum = torch.zeros_like(param) if amsgrad else None
    pointers = [t.data_ptr() for t in (param, m, v)]
    for step_value in range(1, 4):
        grad = torch.full((10, 10), 0.5, device="npu", dtype=grad_dtype)
        step = torch.tensor([step_value], device="npu", dtype=torch.int64)
        cpu_param.grad = torch.full_like(cpu_param, -0.5 if maximize else 0.5)
        optimizer.step()
        result = npu_apply_fused_adamw_v2(
            param, grad, m, v, maximum, step, weight_decay=0.01, amsgrad=amsgrad, maximize=maximize
        )
        assert result is None
        assert step.item() == step_value
        assert pointers == [t.data_ptr() for t in (param, m, v)]
        torch.testing.assert_close(param.cpu().float(), cpu_param.detach(), rtol=0.01, atol=0.005)
        state = optimizer.state[cpu_param]
        for actual, key in ((m, "exp_avg"), (v, "exp_avg_sq")):
            torch.testing.assert_close(actual.cpu().float(), state[key], rtol=0.02, atol=1e-5)
        if amsgrad:
            torch.testing.assert_close(maximum.cpu().float(), state["max_exp_avg_sq"], rtol=0.02, atol=1e-5)


def _moe_reference(tokens, indices, num_out_tokens):
    flat = indices.flatten()
    order = torch.argsort(flat, stable=True)
    topk = 1 if indices.ndim == 1 else indices.shape[1]
    count = flat.numel() if num_out_tokens in (None, 0) else min(num_out_tokens, flat.numel())
    if num_out_tokens is not None and num_out_tokens < 0:
        count = flat.numel() + num_out_tokens
    # CANN stores the inverse permutation for backward and unpermute.
    inverse_order = torch.argsort(order).to(torch.int32)
    return tokens[order[:count] // topk], inverse_order


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("num_out_tokens", [None, 0, 3, -1])
def test_moe_permute_and_gradient(index_dtype, topk, num_out_tokens):
    tokens = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3).requires_grad_()
    indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]], dtype=index_dtype)
    if topk == 1:
        indices = indices[:, 0].contiguous()
    expected, expected_order = _moe_reference(tokens, indices, num_out_tokens)
    x = tokens.detach().npu().requires_grad_()
    actual, order = npu_moe_token_permute(x, indices.npu(), num_out_tokens)
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(order.cpu(), expected_order)
    assert order.numel() == indices.numel()
    assert not order.requires_grad
    dy = torch.ones_like(actual)
    actual.backward(dy)
    expected.backward(dy.cpu())
    torch.testing.assert_close(x.grad.cpu(), tokens.grad)


@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("drop_tokens", [False, True])
@pytest.mark.parametrize("trial", range(3))
def test_moe_unpermute_and_gradients(with_probs, drop_tokens, trial):
    indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]) if with_probs else torch.tensor([0, 4, 2, 1])
    tokens = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    permuted, order = _moe_reference(tokens, indices, 3 if drop_tokens else None)
    permuted = permuted.detach().requires_grad_()
    probs = torch.full((4, 2), 0.5, dtype=torch.bfloat16, requires_grad=True) if with_probs else None
    topk = 2 if with_probs else 1
    positions = torch.argsort(order)[: permuted.shape[0]].long()
    weighted = permuted if probs is None else permuted * probs.flatten()[positions, None]
    expected = torch.zeros(4, 3, dtype=torch.bfloat16).index_add(0, positions // topk, weighted)
    x = permuted.detach().npu().requires_grad_()
    p = probs.detach().npu().requires_grad_() if with_probs else None
    actual = npu_moe_token_unpermute(x, order.npu(), p)
    torch.testing.assert_close(actual.cpu(), expected)
    dy = torch.full_like(actual, trial + 1)
    actual.backward(dy)
    expected.backward(dy.cpu())
    torch.testing.assert_close(x.grad.cpu(), permuted.grad)
    if with_probs:
        torch.testing.assert_close(p.grad.cpu(), probs.grad)


def _rope_reference(x, cos, sin, mode):
    if mode == 0:
        left, right = x.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
    else:
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("trig_grad", [False, True])
def test_rope_modes_and_gradients(dtype, mode, trig_grad):
    torch.manual_seed(42)
    x = torch.randn(2, 4096, 5, 128, dtype=dtype).float().requires_grad_()
    cos = torch.randn(1, 4096, 1, 128, dtype=dtype).float().requires_grad_(trig_grad)
    sin = torch.randn_like(cos).to(dtype).float().requires_grad_(trig_grad)
    expected = _rope_reference(x, cos, sin, mode)
    device_inputs = [t.detach().to(dtype).npu().requires_grad_(t.requires_grad) for t in (x, cos, sin)]
    actual = npu_rotary_position_embedding(*device_inputs, mode=mode)
    torch.testing.assert_close(actual.cpu(), expected.detach().to(dtype), rtol=0.005, atol=0.005)
    dy = torch.ones_like(actual)
    actual.backward(dy)
    expected.backward(dy.cpu().float())
    for value, reference in zip(device_inputs, (x, cos, sin)):
        if reference.requires_grad:
            torch.testing.assert_close(value.grad.cpu(), reference.grad.to(dtype), rtol=0.01, atol=0.03)


def test_rope_invalid_mode():
    x = torch.ones(1, 1, 1, 128, device="npu")
    with pytest.raises(ValueError, match="mode"):
        npu_rotary_position_embedding(x, x, x, mode=2)
