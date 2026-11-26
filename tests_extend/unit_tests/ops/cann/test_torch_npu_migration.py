# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Numerical and mutation contracts for the TorchNPU operator migration."""

import os
import subprocess
import sys

import pytest
import torch
import torch_npu

from mindspeed.ops.gmm import npu_gmm, npu_gmm_v2
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
from mindspeed.ops.quant_gmm import npu_quant_gmm, npu_quant_gmm_v2
from mindspeed.ops.weight_quant_gmm import npu_weight_quant_gmm, npu_weight_quant_gmm_v2

COUNTS = [0, 16, 16, 32]


@pytest.fixture(autouse=True)
def forbid_mindspeed_jit(monkeypatch):
    from mindspeed.op_builder.builder import MindSpeedOpBuilder

    def unexpected_build(*args, **kwargs):
        raise AssertionError("Migrated operators must execute without a MindSpeed JIT extension.")

    monkeypatch.setattr(MindSpeedOpBuilder, 'load', unexpected_build)


@pytest.fixture
def isolated_megatron_case(request):
    # Framework imports patch Tensor APIs and torch.compile process-wide.
    if os.environ.get('MINDSPEED_OPERATOR_STAGE_CHILD') == '1':
        return False
    env = os.environ.copy()
    env['MINDSPEED_OPERATOR_STAGE_CHILD'] = '1'
    command = [sys.executable, '-m', 'pytest', '--import-mode=importlib', '-q', request.node.nodeid]
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=180, check=False, cwd=request.config.rootpath
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


@pytest.fixture(scope='module')
def isolated_graph_suite(request):
    # Other test modules import MegatronAdaptor and patch compile and Tensor APIs.
    if os.environ.get('MINDSPEED_GRAPH_TEST_CHILD') == '1':
        return False
    nodeids = [
        item.nodeid
        for item in request.session.items
        if item.path == request.path
        and (
            item.originalname in ('test_torchair_graph_matches_eager', 'test_torchair_gmm_training_gradients')
            or (item.originalname == 'test_gmm_reads_updated_device_groups' and item.callspec.params['graph'])
        )
    ]
    assert nodeids
    env = os.environ.copy()
    env['MINDSPEED_GRAPH_TEST_CHILD'] = '1'
    command = [sys.executable, '-m', 'pytest', '--import-mode=importlib', '-q', *nodeids]
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=900, check=False, cwd=request.config.rootpath
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def group_values(mode):
    counts = torch.tensor(COUNTS, dtype=torch.int64)
    if mode == 'list':
        return counts.cumsum(0).tolist(), 0
    return (counts if mode == 'counts' else counts.cumsum(0)).npu(), int(mode == 'counts')


def compile_forward(forward):
    import torchair

    torch._dynamo.reset()

    compiled_graphs = []
    npu_backend = torchair.get_npu_backend()

    def backend(gm, inputs):
        compiled_graphs.append(gm)
        return npu_backend(gm, inputs)

    return torch.compile(forward, backend=backend, fullgraph=True), compiled_graphs


def compile_gmm(fn):
    def forward(a, b, g):
        return fn(a, b, group_list=g)

    return compile_forward(forward)


def assert_numerical(actual, expected, dtype):
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 0.0003
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=tolerance, atol=tolerance)


def grouped_reference(x, weight):
    chunks = x.split(COUNTS, dim=0)
    return torch.cat([part.float() @ weight[i].float() for i, part in enumerate(chunks)], dim=0)


def gradient_reference(x, weight, dy):
    xs, dys = x.split(COUNTS, dim=0), dy.split(COUNTS, dim=0)
    dx = torch.cat([part.float() @ weight[i].float().t() for i, part in enumerate(dys)])
    dw = torch.stack([part.float().t() @ dys[i].float() for i, part in enumerate(xs)])
    return dx, dw


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('mode', ['tensor', 'counts', 'list'])
@pytest.mark.parametrize('transpose', [False, True])
def test_gmm_forward_backward(dtype, mode, transpose):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu().requires_grad_()
    w = torch.randn(4, 32, 16, dtype=dtype)
    if transpose:
        w = w.transpose(-1, -2).contiguous().transpose(-1, -2)
    w = w.npu().detach().requires_grad_()
    groups, group_type = group_values(mode)
    fn = npu_gmm_v2 if group_type else npu_gmm
    dy = torch.randn(64, 16, dtype=dtype).npu()
    out = fn(x, w, group_list=groups)
    out.backward(dy)
    expected = grouped_reference(x.detach().cpu(), w.detach().cpu()).to(dtype)
    dx, dw = gradient_reference(x.detach().cpu(), w.detach().cpu(), dy.cpu())
    assert_numerical(out.detach(), expected, dtype)
    assert_numerical(x.grad, dx.to(dtype), dtype)
    assert_numerical(w.grad, dw.to(dtype), dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['tensor', 'counts', 'list'])
def test_gmm_fused_main_gradient(dtype, mode):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu().requires_grad_()
    w = torch.randn(4, 32, 16, dtype=dtype).npu().requires_grad_()
    w.main_grad = torch.randn(4, 32, 16, dtype=torch.float32).npu()
    w.grad_added_to_main_grad = False
    w.zero_out_wgrad = True
    before = w.main_grad.cpu()
    pointer = w.main_grad.data_ptr()
    groups, group_type = group_values(mode)
    dy = torch.randn(64, 16, dtype=dtype).npu()
    fn = npu_gmm_v2 if group_type else npu_gmm
    for _ in range(2):
        fn(x, w, group_list=groups, gemm_fusion=True, original_weight=w).backward(dy)
    _, dw = gradient_reference(x.detach().cpu(), w.detach().cpu(), dy.cpu())
    torch.testing.assert_close(w.main_grad.cpu(), before + 2 * dw, rtol=0.0003, atol=0.0003)
    assert w.main_grad.data_ptr() == pointer
    assert w.grad_added_to_main_grad
    assert torch.count_nonzero(w.grad).item() == 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['tensor', 'counts', 'list'])
@pytest.mark.parametrize('flatten', [False, True])
def test_grouped_add_mutates_original_storage(dtype, mode, flatten):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu()
    dy = torch.randn(64, 16, dtype=dtype).npu()
    grad = torch.ones(4, 32, 16, dtype=torch.float32).npu()
    if flatten:
        grad = grad.view(128, 16)
    pointer = grad.data_ptr()
    groups, group_type = group_values(mode)
    result = npu_groupmatmul_add_fp32(x, dy, groups, grad, group_list_type=group_type)
    _, dw = gradient_reference(x.cpu(), torch.empty(4, 32, 16), dy.cpu())
    torch.testing.assert_close(grad.cpu().view_as(dw), 1 + dw, rtol=0.0003, atol=0.0003)
    assert result is None
    assert grad.data_ptr() == pointer


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_matmul_fp32_accumulation(dtype):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu()
    dy = torch.randn(64, 16, dtype=dtype).npu()
    grad = torch.randn(16, 32, dtype=torch.float32).npu()
    before, pointer = grad.cpu(), grad.data_ptr()
    expected = before + 2 * (dy.cpu().float().t() @ x.cpu().float())
    assert npu_matmul_add_fp32(x, dy, grad) is None
    npu_matmul_add_fp32(x, dy, grad)
    torch.testing.assert_close(grad.cpu(), expected, rtol=0.0003, atol=0.0003)
    assert grad.data_ptr() == pointer


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_matmul_fp16_accumulation(dtype):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu()
    dy = torch.randn(64, 16, dtype=dtype).npu()
    grad = torch.randn(16, 32, dtype=dtype).npu()
    expected = grad.cpu() + (dy.cpu().float().t() @ x.cpu().float()).to(dtype)
    assert npu_matmul_add_fp16(x, dy, grad) is None
    assert_numerical(grad, expected, dtype)


@pytest.mark.parametrize('fn', [npu_matmul_add_fp32, npu_matmul_add_fp16])
def test_matmul_empty_input_is_noop(fn):
    x, dy = torch.empty(0, 32).npu(), torch.empty(0, 16).npu()
    grad = torch.ones(16, 32).npu()
    pointer = grad.data_ptr()
    assert fn(x, dy, grad) is None
    assert torch.equal(grad.cpu(), torch.ones(16, 32))
    assert grad.data_ptr() == pointer


@pytest.mark.parametrize('mode', ['tensor', 'counts'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, None])
def test_quant_gmm(mode, dtype):
    torch.manual_seed(295)
    x = torch.randint(-8, 9, (64, 32), dtype=torch.int8).npu()
    w = torch.randint(-8, 9, (4, 32, 16), dtype=torch.int8).npu()
    output_dtype = torch.bfloat16 if dtype is None else dtype
    scale_dtype = torch.bfloat16 if output_dtype == torch.bfloat16 else torch.float32
    scale = (torch.rand(4, 16, dtype=scale_dtype) * 0.1).npu()
    groups, group_type = group_values(mode)
    fn = npu_quant_gmm_v2 if group_type else npu_quant_gmm
    out = fn(x, w, scale, group_list=groups, output_dtype=dtype)
    scale_rows = torch.repeat_interleave(scale.cpu().float(), torch.tensor(COUNTS), dim=0)
    expected = (grouped_reference(x.cpu(), w.cpu()) * scale_rows).to(output_dtype)
    assert out.dtype == output_dtype
    assert_numerical(out, expected, output_dtype)


@pytest.mark.parametrize('mode', ['tensor', 'counts'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_weight_quant_gmm(mode, dtype):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu()
    w = torch.randint(-8, 9, (4, 32, 16), dtype=torch.int8).npu()
    scale = (torch.rand(4, 16, dtype=dtype) * 0.1).npu()
    offset = torch.rand(4, 16, dtype=dtype).npu()
    groups, group_type = group_values(mode)
    fn = npu_weight_quant_gmm_v2 if group_type else npu_weight_quant_gmm
    out = fn(x, w, scale, antiquant_offset=offset, group_list=groups)
    dequant = ((w.cpu().float() + offset.cpu().float().unsqueeze(1)) * scale.cpu().float().unsqueeze(1)).to(dtype)
    expected = grouped_reference(x.cpu(), dequant).to(dtype)
    assert_numerical(out, expected, dtype)


def test_gmm_rejects_differentiable_bias():
    x, w = torch.ones(64, 32).npu(), torch.ones(4, 32, 16).npu()
    bias = torch.ones(4, 16).npu().requires_grad_()
    groups, _ = group_values('tensor')
    with pytest.raises(ValueError, match='Bias'):
        npu_gmm(x, w, bias=bias, group_list=groups)


def test_gmm_rejects_fusion_without_main_gradient():
    x, w = torch.ones(64, 32).npu(), torch.ones(4, 32, 16).npu()
    groups, _ = group_values('tensor')
    with pytest.raises(ValueError, match='main_grad'):
        npu_gmm(x, w, group_list=groups, gemm_fusion=True)


def test_quant_default_meta_dtype_matches_eager():
    x = torch.empty(64, 32, dtype=torch.int8, device='meta')
    w = torch.empty(4, 32, 16, dtype=torch.int8, device='meta')
    scale = torch.empty(4, 16, dtype=torch.bfloat16, device='meta')
    groups = torch.empty(4, dtype=torch.int64, device='meta')
    result = torch.ops.mindspeed.npu_quant_gmm(x, w, scale, group_list=groups)
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize('kind', ['gmm', 'quant', 'weight_quant'])
def test_torchair_graph_matches_eager(kind, isolated_graph_suite):
    if isolated_graph_suite:
        return
    torch.manual_seed(295)
    groups = torch.tensor(COUNTS, dtype=torch.int64).npu()
    if kind == 'gmm':
        x, w = torch.randn(64, 32, dtype=torch.float16).npu(), torch.randn(4, 32, 16, dtype=torch.float16).npu()

        def forward(a, b, g):
            return npu_gmm_v2(a, b, group_list=g)

    elif kind == 'quant':
        x = torch.randint(-8, 9, (64, 32), dtype=torch.int8).npu()
        w = torch.randint(-8, 9, (4, 32, 16), dtype=torch.int8).npu()
        scale = torch.ones(4, 16, dtype=torch.bfloat16).npu()

        def forward(a, b, g):
            return npu_quant_gmm_v2(a, b, scale, group_list=g)

    else:
        x = torch.randn(64, 32, dtype=torch.float16).npu()
        w = torch.randint(-8, 9, (4, 32, 16), dtype=torch.int8).npu()
        scale = torch.ones(4, 16, dtype=torch.float16).npu()
        offset = torch.zeros_like(scale)

        def forward(a, b, g):
            return npu_weight_quant_gmm_v2(a, b, scale, antiquant_offset=offset, group_list=g)

    compiled, compiled_graphs = compile_forward(forward)
    with torch.no_grad():
        expected = forward(x, w, groups)
        actual = compiled(x, w, groups)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.01, atol=0.01)
    assert len(compiled_graphs) == 1


@pytest.mark.parametrize('mode', ['tensor', 'counts'])
def test_quant_per_token_scale_and_bias(mode):
    torch.manual_seed(295)
    x = torch.randint(-8, 9, (64, 32), dtype=torch.int8).npu()
    w = torch.randint(-8, 9, (4, 32, 16), dtype=torch.int8).npu()
    scale = torch.full((4, 16), 0.05, dtype=torch.float32).npu()
    per_token = torch.rand(64, dtype=torch.float32).npu()
    bias = torch.randint(-4, 5, (4, 16), dtype=torch.int32).npu()
    groups, group_type = group_values(mode)
    fn = npu_quant_gmm_v2 if group_type else npu_quant_gmm
    actual = fn(x, w, scale, per_token_scale=per_token, bias=bias, group_list=groups, output_dtype=torch.float16)
    bias_rows = torch.repeat_interleave(bias.cpu().float(), torch.tensor(COUNTS), dim=0)
    expected = ((grouped_reference(x.cpu(), w.cpu()) + bias_rows) * 0.05 * per_token.cpu().unsqueeze(1)).half()
    assert_numerical(actual, expected, torch.float16)


@pytest.mark.parametrize('mode', ['tensor', 'counts'])
def test_quant_int8_output(mode):
    torch.manual_seed(295)
    x = torch.randint(-8, 9, (64, 32), dtype=torch.int8).npu()
    w = torch.randint(-8, 9, (4, 32, 16), dtype=torch.int8).npu()
    x[:32] = 8
    w[1] = 8
    w[2] = -8
    scale_float = torch.full((4, 16), 0.125, dtype=torch.float32).npu()
    scale = torch_npu.npu_trans_quant_param(scale_float.reshape(-1)).reshape_as(scale_float)
    groups, group_type = group_values(mode)
    fn = npu_quant_gmm_v2 if group_type else npu_quant_gmm
    actual = fn(x, w, scale, group_list=groups, output_dtype=torch.int8)
    expected = (grouped_reference(x.cpu(), w.cpu()) * 0.125).round().clamp(-128, 127).to(torch.int8)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
def test_megatron_bf16_gmm_operator_stages(dtype, mode, isolated_megatron_case):
    if isolated_megatron_case:
        return
    from mindspeed.core.transformer.moe.grouped_matmul_util import BF16GMMFunction

    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu()
    w = torch.randn(4, 32, 16, dtype=dtype).npu()
    dy = torch.randn(64, 16, dtype=dtype).npu()
    groups, group_type = group_values(mode)
    actual = BF16GMMFunction.op_forward(None, x, w, groups, group_type)[0]
    dx = BF16GMMFunction.op_dx(None, dy, w, groups, group_type)[0]
    dw = BF16GMMFunction.op_dw(None, x, dy, groups, group_type)[0]
    expected = grouped_reference(x.cpu(), w.cpu()).to(dtype)
    expected_dx, expected_dw = gradient_reference(x.cpu(), w.cpu(), dy.cpu())
    assert_numerical(actual, expected, dtype)
    assert_numerical(dx, expected_dx.to(dtype), dtype)
    assert_numerical(dw, expected_dw.to(dtype), dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
def test_delayed_weight_gradient_uses_group_representation(monkeypatch, dtype, mode, isolated_megatron_case):
    if isolated_megatron_case:
        return
    from types import SimpleNamespace
    from mindspeed.core.transformer.moe import grouped_matmul_util
    from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules import weight_grad_store

    args = SimpleNamespace(fp8=False, use_gmm_fp8=False, gemm_gradient_accumulation_fusion=False)
    monkeypatch.setattr(grouped_matmul_util, 'get_args', lambda: args)
    monkeypatch.setattr(weight_grad_store, 'get_args', lambda: args)
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu()
    dy = torch.randn(64, 16, dtype=dtype).npu()
    weight = torch.empty(128, 16, dtype=dtype).npu()
    weight.gmm_weight = True
    weight.main_grad = torch.ones(128, 16, dtype=torch.float32).npu()
    pointer = weight.main_grad.data_ptr()
    _, expected = gradient_reference(x.cpu(), torch.empty(4, 32, 16), dy.cpu())
    expected = 1 + expected.to(dtype).float().view(128, 16)
    groups, group_type = group_values(mode)
    cache = ([None, x, groups, group_type, (4, 32, 16)], dy, weight, False, False)
    assert weight_grad_store.WeightGradStore.overlap_matmul(cache) is None
    assert_numerical(weight.main_grad, expected, dtype)
    assert weight.main_grad.data_ptr() == pointer


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
@pytest.mark.parametrize('requires_x,requires_w', [(True, False), (False, True)])
def test_gmm_single_requested_gradient(dtype, mode, requires_x, requires_w):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu().requires_grad_(requires_x)
    w = torch.randn(4, 32, 16, dtype=dtype).npu().requires_grad_(requires_w)
    dy = torch.randn(64, 16, dtype=dtype).npu()
    groups, group_type = group_values(mode)
    fn = npu_gmm_v2 if group_type else npu_gmm
    dx, dw = gradient_reference(x.detach().cpu(), w.detach().cpu(), dy.cpu())
    fn(x, w, group_list=groups).backward(dy)
    if requires_x:
        assert_numerical(x.grad, dx.to(dtype), dtype)
        assert w.grad is None
    else:
        assert x.grad is None
        assert_numerical(w.grad, dw.to(dtype), dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
def test_gmm_fusion_updates_main_gradient_for_frozen_weight(dtype, mode):
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu().requires_grad_()
    w = torch.randn(4, 32, 16, dtype=dtype).npu()
    w.main_grad = torch.ones(4, 32, 16, dtype=torch.float32).npu()
    groups, group_type = group_values(mode)
    fn = npu_gmm_v2 if group_type else npu_gmm
    dy = torch.randn(64, 16, dtype=dtype).npu()
    dx, dw = gradient_reference(x.detach().cpu(), w.cpu(), dy.cpu())
    fn(x, w, group_list=groups, gemm_fusion=True, original_weight=w).backward(dy)
    assert_numerical(x.grad, dx.to(dtype), dtype)
    torch.testing.assert_close(w.main_grad.cpu(), 1 + dw, rtol=0.0003, atol=0.0003)
    assert w.grad is None


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
@pytest.mark.parametrize('graph', [False, True])
def test_gmm_reads_updated_device_groups(dtype, mode, graph, request):
    if graph and request.getfixturevalue('isolated_graph_suite'):
        return
    torch.manual_seed(295)
    x = torch.randn(67, 31, dtype=dtype).npu().requires_grad_()
    w = torch.randn(4, 17, 31, dtype=dtype).npu().transpose(-1, -2).detach().requires_grad_()
    dy = torch.randn(67, 17, dtype=dtype).npu()
    groups = torch.empty(4, dtype=torch.int64).npu()
    pointer = groups.data_ptr()
    fn = npu_gmm_v2 if mode == 'counts' else npu_gmm
    call, compiled_graphs = compile_gmm(fn) if graph else (lambda a, b, g: fn(a, b, group_list=g), [])
    for values in ([0, 17, 16, 34], [19, 0, 33, 15], [16, 17, 17, 17]):
        counts = torch.tensor(values, dtype=torch.int64)
        groups.copy_((counts if mode == 'counts' else counts.cumsum(0)).npu())
        x.grad = w.grad = None
        xs, dys = x.detach().cpu().split(values), dy.cpu().split(values)
        wc = w.detach().cpu().float()
        expected = torch.cat([part.float() @ wc[i] for i, part in enumerate(xs)])
        dx = torch.cat([part.float() @ wc[i].t() for i, part in enumerate(dys)])
        dw = torch.stack([part.float().t() @ dys[i].float() for i, part in enumerate(xs)])
        out = call(x, w, groups)
        out.backward(dy)
        assert_numerical(out.detach(), expected.to(dtype), dtype)
        assert_numerical(x.grad, dx.to(dtype), dtype)
        assert_numerical(w.grad, dw.to(dtype), dtype)
        assert groups.data_ptr() == pointer

    if graph:
        assert len(compiled_graphs) == 1


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
def test_gmm_split_k_two_dimensional_inputs(dtype, mode):
    torch.manual_seed(295)
    x = torch.randn(67, 31, dtype=dtype).npu().transpose(0, 1)
    w = torch.randn(67, 17, dtype=dtype).npu()
    groups = torch.tensor([0, 17, 16, 34], dtype=torch.int64).npu()
    fn = npu_gmm_v2 if mode == 'counts' else npu_gmm
    if mode != 'counts':
        groups = groups.cumsum(0)
    actual = fn(x, w, group_list=groups, group_type=2)
    xs, ws = x.cpu().split([0, 17, 16, 34], dim=1), w.cpu().split([0, 17, 16, 34])
    expected = torch.stack([a.float() @ b.float() for a, b in zip(xs, ws)])
    assert_numerical(actual, expected.to(dtype), dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('mode', ['tensor', 'counts'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('requested', ['both', 'x', 'weight'])
def test_torchair_gmm_training_gradients(dtype, mode, transpose, requested, isolated_graph_suite):
    if isolated_graph_suite:
        return
    torch.manual_seed(295)
    x = torch.randn(64, 32, dtype=dtype).npu().requires_grad_(requested != 'weight')
    w = torch.randn(4, 32, 16, dtype=dtype)
    if transpose:
        w = w.transpose(-1, -2).contiguous().transpose(-1, -2)
    w = w.npu().detach().requires_grad_(requested != 'x')
    dy = torch.randn(64, 16, dtype=dtype).npu()
    groups, group_type = group_values(mode)
    fn = npu_gmm_v2 if group_type else npu_gmm

    compiled, compiled_graphs = compile_gmm(fn)
    actual = compiled(x, w, groups)
    actual.backward(dy)
    expected = grouped_reference(x.detach().cpu(), w.detach().cpu())
    dx, dw = gradient_reference(x.detach().cpu(), w.detach().cpu(), dy.cpu())
    assert_numerical(actual.detach(), expected.to(dtype), dtype)
    if requested != 'weight':
        assert_numerical(x.grad, dx.to(dtype), dtype)
    else:
        assert x.grad is None
    if requested != 'x':
        assert_numerical(w.grad, dw.to(dtype), dtype)
    else:
        assert w.grad is None

    assert len(compiled_graphs) == 1
