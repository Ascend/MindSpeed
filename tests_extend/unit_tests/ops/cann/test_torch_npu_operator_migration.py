# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import pytest
import torch

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
@pytest.mark.parametrize("trial", range(3))
def test_moe_unpermute_and_gradients(with_probs, trial):
    # Token-drop coverage (num_out_tokens < num_tokens * topk, i.e. a short
    # permuted_tokens) is intentionally omitted: Megatron does not use this
    # operator on the token-drop path, and the Ascend 950 lowering of
    # aclnnMoeTokenUnpermuteGrad still rejects a short expanded_x. Restore it
    # together with the CANN fix.
    indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]) if with_probs else torch.tensor([0, 4, 2, 1])
    tokens = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    permuted, order = _moe_reference(tokens, indices, None)
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
