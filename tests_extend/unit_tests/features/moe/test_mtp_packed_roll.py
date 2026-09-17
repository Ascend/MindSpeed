# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Exact CPU checks for packed MTP movement, isolated from NPU initialization.

Run with pytest --noconftest to avoid the distributed training test harness.
These tests execute the production wrapper with real PyTorch tensors. They do
not measure accelerator performance or exercise CP collectives.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _mindspeed_source_root():
    # CI may copy tests_extend into Megatron-LM. Resolve the active package
    # without importing MindSpeed's feature/NPU initialization modules.
    spec = importlib.util.find_spec('mindspeed')
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parent.parent
    # Also support standalone CPU tests in an uninstalled source checkout.
    root = Path(__file__).resolve().parents[4]
    if (root / 'mindspeed/__init__.py').is_file():
        return root
    raise RuntimeError('Cannot locate MindSpeed sources; install the matching version or set PYTHONPATH to its root.')


ROOT = _mindspeed_source_root()


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


roll_module = _load_file('_mtp_roll_test', ROOT / 'mindspeed/core/transformer/multi_token_prediction.py')


def _reference(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Megatron 0.18's original CP=1 clone-and-slice algorithm."""
    assert cp_group is None or cp_group.size() == 1
    assert dims == -1 or dims == tensor.dim() - 1
    assert shifts == -1
    boundaries = packed_seq_params.cu_seqlens_q
    assert boundaries is not None
    result = tensor.clone()
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        shifted = torch.roll(tensor[..., start:end], shifts=shifts, dims=dims)
        shifted[..., shifts:] = 0
        result[..., start:end] = shifted
    return result, result.sum()


optimized = roll_module.roll_tensor_packed_seq_wrapper(_reference)


def _assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    actual_bytes = actual.contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def _input(dtype, layout, length=16):
    generator = torch.Generator().manual_seed(417)
    tensor = torch.randn(3, length, generator=generator).to(dtype)
    if dtype in (torch.int32, torch.int64):
        tensor = torch.arange(3 * length, dtype=dtype).reshape(3, length)
    elif dtype == torch.bool:
        tensor = tensor > 0
    if layout == 'vector':
        return tensor[0]
    if layout == 'transposed':
        return tensor.t().contiguous().t()
    if layout == 'strided':
        storage = torch.zeros(3, 2 * length, dtype=dtype)
        storage[:, ::2] = tensor
        return storage[:, ::2]
    if layout == 'rank3':
        return tensor.unsqueeze(0).expand(2, -1, -1)
    return tensor


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.bool, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('layout', ['vector', 'matrix', 'transposed', 'strided', 'rank3'])
@pytest.mark.parametrize(
    'boundaries',
    [
        [0, 16],
        [0, 1, 7, 16],
        [0, 0, 4, 4, 16, 16],
        [3, 6, 16],
        [3, 6, 11],
        [0, 0, 0],
        [5, 5],
        [16],
        [],
        [0, 7, 16, 32, 48],
        [3, 25, 30],
        [20, 32],
    ],
)
def test_output_sum_layout_and_input_are_unchanged(dtype, layout, boundaries):
    tensor = _input(dtype, layout)
    original = tensor.clone()
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(boundaries, dtype=torch.int32))
    expected, expected_sum = _reference(tensor, -1, -1, packed)
    actual, actual_sum = optimized(tensor, -1, tensor.dim() - 1, packed)
    _assert_bits_equal(actual, expected)
    _assert_bits_equal(actual_sum, expected_sum)
    assert actual.stride() == expected.stride()
    _assert_bits_equal(tensor, original)
    assert actual.data_ptr() != tensor.data_ptr()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('layout', ['matrix', 'transposed', 'strided'])
@pytest.mark.parametrize('boundaries', [[0, 2, 2, 7, 16], [3, 6, 11], [5, 5], []])
def test_input_gradient_is_bitwise_unchanged(dtype, layout, boundaries):
    tensor = _input(dtype, layout).requires_grad_()
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(boundaries, dtype=torch.int64))
    weights = torch.randn(tensor.shape, generator=torch.Generator().manual_seed(51)).to(dtype)
    expected, expected_sum = _reference(tensor, -1, -1, packed)
    actual, actual_sum = optimized(tensor, -1, -1, packed)
    expected_grad = torch.autograd.grad((expected, expected_sum), tensor, (weights, torch.ones_like(expected_sum)))[0]
    actual_grad = torch.autograd.grad((actual, actual_sum), tensor, (weights, torch.ones_like(actual_sum)))[0]
    _assert_bits_equal(actual_grad, expected_grad)


def test_masked_loss_and_logits_gradient_match_across_mtp_depths():
    labels = torch.arange(16).reshape(1, -1) % 7
    mask = torch.tensor([[1, 1, 0, 1] * 4], dtype=torch.float32)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 4, 9, 16]))
    logits = torch.randn(1, 16, 7, generator=torch.Generator().manual_seed(43), requires_grad=True)
    old_labels, old_mask = labels, mask
    new_labels, new_mask = labels, mask
    for _ in range(3):
        old_labels, _ = _reference(old_labels, -1, -1, packed)
        old_mask, old_count = _reference(old_mask, -1, -1, packed)
        new_labels, _ = optimized(new_labels, -1, -1, packed)
        new_mask, new_count = optimized(new_mask, -1, -1, packed)
        old_loss = (
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), old_labels.flatten(), reduction='none')
            * old_mask.flatten()
        ).sum() / old_count
        new_loss = (
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), new_labels.flatten(), reduction='none')
            * new_mask.flatten()
        ).sum() / new_count
        _assert_bits_equal(new_loss, old_loss)
        old_grad = torch.autograd.grad(old_loss, logits)[0]
        new_grad = torch.autograd.grad(new_loss, logits)[0]
        _assert_bits_equal(new_grad, old_grad)


def test_nonfinite_values_signed_zeros_and_empty_tensor():
    tensor = torch.tensor([[-0.0, float('nan'), 0.0, float('inf'), -float('inf'), -0.0, 9.0]])
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 6, 7]))
    actual = optimized(tensor, -1, -1, packed)
    expected = _reference(tensor, -1, -1, packed)
    for result, reference in zip(actual, expected):
        _assert_bits_equal(result, reference)
    empty = torch.empty(2, 0)
    packed.cu_seqlens_q = torch.tensor([0, 0, 0])
    for result, reference in zip(optimized(empty, -1, -1, packed), _reference(empty, -1, -1, packed)):
        _assert_bits_equal(result, reference)


def test_interleaved_microbatches_and_mutated_boundaries():
    tensor = _input(torch.float32, 'matrix')
    first = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 8, 16]))
    second = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 6, 13, 16]))
    for packed in (first, second, first):
        _assert_bits_equal(optimized(tensor, -1, -1, packed)[0], _reference(tensor, -1, -1, packed)[0])
    first.cu_seqlens_q[1] = 5
    _assert_bits_equal(optimized(tensor, -1, -1, first)[0], _reference(tensor, -1, -1, first)[0])


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_roll_and_dispatch_counts_do_not_grow_with_document_count():
    counts = []
    for documents in (2, 16, 128):
        tensor = torch.arange(documents * 4).reshape(1, -1)
        packed = SimpleNamespace(cu_seqlens_q=torch.arange(0, documents * 4 + 1, 4))
        with _OpCounter() as counter:
            optimized(tensor, -1, -1, packed)
        assert counter.ops.count('aten.roll.default') == 1
        assert 'aten._local_scalar_dense.default' not in counter.ops
        counts.append(len(counter.ops))
    assert len(set(counts)) == 1


@pytest.mark.parametrize('case', ['cp', 'shift', 'dim', 'missing', 'list', 'dtype', 'rank'])
def test_other_paths_delegate_to_original(case):
    tensor = torch.ones(2, 4)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 4]))
    cp = SimpleNamespace(size=lambda: 2) if case == 'cp' else None
    shifts = -2 if case == 'shift' else -1
    dims = 0 if case == 'dim' else -1
    if case == 'missing':
        packed.cu_seqlens_q = None
    elif case == 'list':
        packed.cu_seqlens_q = [0, 4]
    elif case == 'dtype':
        packed.cu_seqlens_q = torch.tensor([0.0, 4.0])
    elif case == 'rank':
        packed.cu_seqlens_q = torch.tensor([[0, 4]])
    sentinel = object()
    original = Mock(return_value=sentinel)
    wrapper = roll_module.roll_tensor_packed_seq_wrapper(original)
    assert wrapper(tensor, shifts, dims, packed, cp) is sentinel
    original.assert_called_once_with(tensor, shifts, dims, packed, cp)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('mtp_layers', [None, 0, 1, 2])
@pytest.mark.parametrize('mtp_first', [False, True])
@pytest.mark.parametrize('has_helper', [False, True])
def test_mtp_patch_is_registered_once_independent_of_fb(monkeypatch, enabled, mtp_layers, mtp_first, has_helper):
    class BaseFeature:
        def __init__(self, name):
            self.feature_name = name.replace('-', '_')

    def stub_module(name, **exports):
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)

    stub_module('mindspeed.features_manager.feature', MindSpeedFeature=BaseFeature)
    package = 'mindspeed.core.transformer.moe.moe_feature.fb_overlap'
    stub_module(
        package,
        **{
            name: Mock()
            for name in (
                'linear_backward_wgrad_detach',
                'transformer_block_fb_overlap_init_wrapper',
                'mtp_block_fb_overlap_forward_wrapper',
                'dualpipev_fb_overlap_mtp_layer_forward',
            )
        },
    )
    stub_module(
        package + '.adaptor',
        **{
            name: Mock()
            for name in (
                '_make_backward_post_hook',
                'fb_overlap_ddp_init_wrapper',
                'get_moe_module_spec_wrapper',
                'get_forward_backward_func_vpp_overlap_wrapper',
            )
        },
    )
    monkeypatch.setitem(sys.modules, 'mindspeed.core.transformer.multi_token_prediction', roll_module)
    fb_module = _load_file('_fb_feature_test', ROOT / 'mindspeed/features_manager/moe/fb_overlap.py')
    mtp_module = _load_file(
        '_mtp_feature_test', ROOT / 'mindspeed/features_manager/transformer/multi_token_prediction.py'
    )
    native_module = SimpleNamespace()
    if has_helper:
        native_module._roll_tensor_packed_seq = _reference
    importer = Mock(return_value=native_module)
    monkeypatch.setattr(mtp_module, 'import_module', importer)
    recorder = Mock()
    args = SimpleNamespace(moe_fb_overlap=enabled, mtp_num_layers=mtp_layers, pipeline_model_parallel_size=2)
    features = [mtp_module.MultiTokenPredictionFeature(), fb_module.MoEFwdBwdOverlapFeature()]
    for feature in features if mtp_first else reversed(features):
        feature.register_patches(recorder, args)
    target = 'megatron.core.transformer.multi_token_prediction._roll_tensor_packed_seq'
    matches = [call.args for call in recorder.register_patch.call_args_list if call.args[0] == target]
    assert matches == ([(target, roll_module.roll_tensor_packed_seq_wrapper)] if mtp_layers and has_helper else [])
    fb_target = 'megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward'
    fb_matches = [call for call in recorder.register_patch.call_args_list if call.args[0] == fb_target]
    assert len(fb_matches) == int(bool(enabled and mtp_layers))
    assert importer.call_count == int(bool(mtp_layers))


@pytest.mark.parametrize(
    'missing_name,skip',
    [
        ('megatron', True),
        ('megatron.core.transformer.multi_token_prediction', True),
        ('transformer_engine', False),
        (None, False),
    ],
)
def test_mtp_feature_skips_absent_core_but_preserves_dependency_errors(monkeypatch, missing_name, skip):
    base = ModuleType('mindspeed.features_manager.feature')
    base.MindSpeedFeature = object
    monkeypatch.setitem(sys.modules, base.__name__, base)
    module = _load_file(
        '_mtp_compatibility_test', ROOT / 'mindspeed/features_manager/transformer/multi_token_prediction.py'
    )
    error = ModuleNotFoundError('missing module', name=missing_name)
    monkeypatch.setattr(module, 'import_module', Mock(side_effect=error))
    feature = object.__new__(module.MultiTokenPredictionFeature)
    recorder = Mock()
    if skip:
        feature.register_patches(recorder, SimpleNamespace(mtp_num_layers=1))
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            feature.register_patches(recorder, SimpleNamespace(mtp_num_layers=1))
        assert caught.value is error
    recorder.register_patch.assert_not_called()
    # A core without the MTP argument must not import its optional MTP module.
    module.import_module.reset_mock()
    feature.register_patches(recorder, SimpleNamespace())
    module.import_module.assert_not_called()


def test_mtp_feature_is_in_transformer_catalog():
    path = ROOT / 'mindspeed/features_manager/__init__.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    builder = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'add_transformer_features'
    )
    constructors = {
        node.func.id for node in ast.walk(builder) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    sentinel = object()
    namespace = {name: (lambda: object()) for name in constructors}
    namespace.update(List=list, MindSpeedFeature=object, MultiTokenPredictionFeature=lambda: sentinel)
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
    features = []
    namespace['add_transformer_features'](features)
    assert features.count(sentinel) == 1


@pytest.mark.skipif(importlib.util.find_spec('torch_npu') is None, reason='requires torch_npu and NPU hardware')
@pytest.mark.parametrize('dtype', [torch.int64, torch.bfloat16, torch.float32])
def test_npu_output_sum_gradient_and_no_scalar_reads(dtype):
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip('NPU hardware unavailable')
    tensor = _input(dtype, 'matrix').to('npu')
    if tensor.is_floating_point():
        tensor.requires_grad_()
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([3, 6, 6, 16, 32], dtype=torch.int32, device='npu'))
    expected, expected_sum = _reference(tensor, -1, -1, packed)
    with _OpCounter() as counter:
        actual, actual_sum = optimized(tensor, -1, -1, packed)
    assert counter.ops.count('aten.roll.default') == 1
    assert 'aten._local_scalar_dense.default' not in counter.ops
    _assert_bits_equal(actual.cpu(), expected.cpu())
    _assert_bits_equal(actual_sum.cpu(), expected_sum.cpu())
    if tensor.requires_grad:
        weights = torch.arange(tensor.numel(), device='npu', dtype=dtype).reshape(tensor.shape)
        actual_grad = torch.autograd.grad((actual, actual_sum), tensor, (weights, torch.ones_like(actual_sum)))[0]
        expected_grad = torch.autograd.grad((expected, expected_sum), tensor, (weights, torch.ones_like(expected_sum)))[
            0
        ]
        _assert_bits_equal(actual_grad.cpu(), expected_grad.cpu())
