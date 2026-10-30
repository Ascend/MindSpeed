"""CPU regression checks; run with pytest --noconftest.

Only loss/gradient comparisons require the core012 forward/autoscaler source.
Set MEGATRON_CORE_SOURCE in CI to require those comparisons; an invalid explicit
path fails instead of skipping. Other checks run without any Megatron checkout.
Small trainable modules replace embedding/transformer/output layers; this does not
exercise NPU attention, distributed collectives or the FB-overlap scheduler.
"""

import ast
import copy
import os
from functools import wraps
from inspect import signature
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import mindspeed

# The tests can live in Megatron while MindSpeed is installed from another tree.
ROOT = Path(mindspeed.__file__).resolve().parent.parent
TEST_ROOT = Path(__file__).resolve().parents[4]
FB = ROOT / 'mindspeed/core/transformer/moe/moe_feature/fb_overlap'


def extract(path, name, namespace, parent=None):
    body = ast.parse(path.read_text(encoding='utf-8')).body
    if parent:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == parent)
    node = next(node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    unit = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(unit)
    # Run the real source bodies without initializing the NPU training stack.
    exec(compile(unit, str(path), 'exec'), namespace)  # noqa: S102  # pylint: disable=exec-used
    return namespace[name]


@pytest.fixture(name='implementation')
def build_implementation():
    logs = []
    namespace = {
        'torch': torch,
        'wraps': wraps,
        'signature': signature,
        'parallel_state': SimpleNamespace(get_tensor_and_context_parallel_group=lambda: None),
        'MTPLossLoggingHelper': SimpleNamespace(
            save_loss_to_tracker=lambda loss, *args, **kwargs: logs.append(loss.detach().clone())
        ),
    }
    extract(FB / 'packed_mtp.py', 'roll_tensor_packed', namespace)
    packed_forward = extract(FB / 'packed_mtp.py', 'mtp_block_packed_forward', namespace)
    wrapper = extract(FB / 'transformer_block.py', 'mtp_block_fb_overlap_forward_wrapper', namespace)
    return SimpleNamespace(env=namespace, logs=logs, forward=packed_forward, wrapper=wrapper)


@pytest.fixture(name='core_implementation')
def build_core_implementation(implementation):
    relative = Path('megatron/core/transformer/multi_token_prediction.py')
    configured = os.environ.get('MEGATRON_CORE_SOURCE')
    if configured:
        native = Path(configured) / relative
        if not native.is_file():
            pytest.fail(f'MEGATRON_CORE_SOURCE does not contain {relative}: {configured}')
    else:
        # Support tests copied into Megatron as well as the local multi-repo layout.
        candidates = [
            TEST_ROOT / relative,
            ROOT / 'Megatron-LM' / relative,
            ROOT.parent / 'Megatron-LM-core_v0.12.1' / relative,
        ]
        native = next((path for path in candidates if path.is_file()), None)
        if native is None:
            pytest.skip('Native loss/gradient comparison requires MEGATRON_CORE_SOURCE pointing to core_v0.12.1')
    extract(native, 'MTPLossAutoScaler', implementation.env)
    extract(native, 'roll_tensor', implementation.env)
    implementation.native = extract(native, 'forward', implementation.env, 'MultiTokenPredictionBlock')
    return implementation


def reference_roll(tensor, packed):
    cu = [0, *packed.cu_seqlens_q.tolist()]
    flat = tensor.reshape(-1)
    result = flat.clone()
    for start, end in pairwise(cu):
        if start < end:
            result[start : end - 1] = flat[start + 1 : end]
            result[end - 1] = 0
    result = result.reshape_as(tensor)
    return result, result.sum()


@pytest.mark.parametrize('endpoints', [[6], [3, 6], [0, 3, 3, 6], [1, 2, 3, 4, 5, 6]])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('endpoint_format', [False, True])
def test_packed_shifts_and_shared_boundaries(implementation, endpoints, batch_size, endpoint_format):
    points = [end + row * 6 for row in range(batch_size) for end in endpoints]
    cu = torch.tensor(points if endpoint_format else [0, *points])
    packed = SimpleNamespace(cu_seqlens_q=cu, cu_seqlens_kv=cu)
    before = cu.clone()
    for tensor in (torch.arange(batch_size * 6).reshape(batch_size, 6), torch.ones(batch_size, 6)):
        original = tensor.clone()
        actual = implementation.env['roll_tensor_packed'](tensor, packed)
        expected, _ = reference_roll(tensor, packed)
        assert torch.equal(actual, expected)
        assert torch.equal(tensor, original)
    assert packed.cu_seqlens_q is cu and packed.cu_seqlens_kv is cu
    assert torch.equal(cu, before)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        self.packed_seen = []

    def forward(self, decoder_input, hidden_states, packed_seq_params, **kwargs):
        self.packed_seen.append(packed_seq_params)
        return torch.tanh(self.proj(decoder_input) + hidden_states)


def run_forward(implementation, forward, modules, packed, depth, per_token, mask):
    embedding, layers, head = modules
    block = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1, mtp_num_layers=depth, calculate_per_token_loss=per_token),
        layers=layers,
        training=True,
        mtp_loss_scaling_factor=0.3,
    )
    hidden = torch.arange(24, dtype=torch.float64).reshape(6, 1, 4).div(24).requires_grad_()
    seen_inputs, seen_labels, seen_positions = [], [], []
    position_ids = torch.arange(6).unsqueeze(0)

    def embed(input_ids, position_ids):
        seen_inputs.append(input_ids.clone())
        seen_positions.append(position_ids.clone())
        return embedding(input_ids).transpose(0, 1)

    def loss(labels, logits):
        seen_labels.append(labels.clone())
        return torch.nn.functional.cross_entropy(logits.permute(1, 2, 0), labels, reduction='none')

    implementation.logs.clear()
    result = forward(
        block,
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        position_ids=position_ids,
        hidden_states=hidden,
        attention_mask=None,
        labels=torch.tensor([[2, 3, 4, 5, 6, 7]]),
        packed_seq_params=packed,
        loss_mask=mask,
        embedding=embed,
        output_layer=lambda value, **kwargs: (head(value), None),
        compute_language_model_loss=loss,
    )
    result.square().sum().backward()
    grads = [hidden.grad] + [p.grad for module in modules for p in module.parameters()]
    for layer in layers:
        assert layer.packed_seen[-1] is packed
    assert all(torch.equal(position, position_ids) for position in seen_positions)
    return result.detach(), grads, list(implementation.logs), seen_inputs, seen_labels


@pytest.mark.parametrize('depth', [1, 2])
@pytest.mark.parametrize('per_token', [False, True])
@pytest.mark.parametrize('with_mask', [False, True])
def test_wrapper_loss_and_gradients_match_core012_segment_reference(core_implementation, depth, per_token, with_mask):
    implementation = core_implementation
    torch.manual_seed(17)
    modules = (
        torch.nn.Embedding(8, 4, dtype=torch.float64),
        torch.nn.ModuleList([ToyLayer() for _ in range(depth)]),
        torch.nn.Linear(4, 8, bias=False, dtype=torch.float64),
    )
    cu = torch.tensor([3, 6])
    packed = SimpleNamespace(cu_seqlens_q=cu, cu_seqlens_kv=cu)
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0, 1.0]]) if with_mask else None
    actual = run_forward(
        implementation, implementation.wrapper(implementation.native), modules, packed, depth, per_token, mask
    )
    # Execute core012's actual Block.forward with only its Roll dependency changed
    # to the independent per-segment reference, including its real loss autoscaler.
    implementation.env['roll_tensor'] = lambda tensor, **kwargs: reference_roll(tensor, packed)
    expected = run_forward(
        implementation, implementation.native, copy.deepcopy(modules), packed, depth, per_token, mask
    )
    assert torch.equal(actual[0], expected[0])
    for actual_values, expected_values in zip(actual[1:], expected[1:]):
        for value, reference in zip(actual_values, expected_values):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert cu.tolist() == [3, 6]


def test_all_masked_targets_have_finite_zero_auxiliary_gradient(core_implementation):
    implementation = core_implementation
    modules = (
        torch.nn.Embedding(8, 4, dtype=torch.float64),
        torch.nn.ModuleList([ToyLayer(), ToyLayer()]),
        torch.nn.Linear(4, 8, bias=False, dtype=torch.float64),
    )
    packed = SimpleNamespace(cu_seqlens_q=torch.arange(1, 7))
    result = run_forward(implementation, implementation.forward, modules, packed, 2, False, None)
    assert all(torch.isfinite(grad).all() for grad in result[1])
    assert all(torch.count_nonzero(grad) == 0 for grad in result[1][1:])
    assert all(loss == 0 for loss in result[2])


def test_nonpacked_delegates_and_cp_packed_is_rejected(implementation):
    calls = []

    def native(self, **kwargs):
        calls.append(kwargs)
        return 'native'

    assert (
        implementation.wrapper(native)(
            None, None, None, None, None, extra_block_kwargs={'bwd_block_graphs': None, 'keep': 1}
        )
        == 'native'
    )
    assert calls[0]['extra_block_kwargs'] == {'keep': 1}
    with pytest.raises(NotImplementedError, match='context_parallel_size=1'):
        implementation.forward(SimpleNamespace(config=SimpleNamespace(context_parallel_size=2)), None, None, None, None)


def test_gpt_endpoints_work_without_changing_attention_metadata(implementation):
    cu = torch.tensor([3, 6])
    env = implementation.env
    env.update(
        get_actual_seq_len=lambda: cu,
        PackedSeqParams=SimpleNamespace,
        compute_qkv_index=lambda value: (None, None),
        get_position_ids=lambda: None,
    )
    wrapper = extract(
        ROOT / 'mindspeed/core/transformer/flash_attention/reset_attention_mask/adaptor.py', 'gpt_forward_wrapper', env
    )
    packed = wrapper(lambda self, packed_seq_params: packed_seq_params)(None)
    assert packed.cu_seqlens_q is cu and packed.cu_seqlens_kv is cu
    mask = env['roll_tensor_packed'](torch.ones(1, 6), packed)
    assert mask.tolist() == [[1, 1, 0, 1, 1, 0]] and mask.sum() == 4


class RollCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_roll_count_is_constant_without_scalar_reads(implementation):
    counts = []
    for documents in (1, 32, 128):
        packed = SimpleNamespace(cu_seqlens_q=torch.arange(4, documents * 4 + 1, 4))
        with RollCounter() as counter:
            implementation.env['roll_tensor_packed'](torch.ones(1, documents * 4), packed)
        assert counter.ops.count('aten.roll.default') == 1
        assert 'aten.sum.default' not in counter.ops
        assert 'aten._local_scalar_dense.default' not in counter.ops
        counts.append(len(counter.ops))
    assert len(set(counts)) == 1


@pytest.mark.parametrize('parameter', ['pre_process', 'post_process'])
def test_extended_forward_is_preserved_nonpacked_and_rejected_packed(implementation, parameter):
    calls = []

    def with_pre_process(self, pre_process=None, **kwargs):
        calls.append(pre_process)
        return 'extended'

    def with_post_process(self, post_process=None, **kwargs):
        calls.append(post_process)
        return 'extended'

    # Exercise either extension independently, including a decorated callable.
    extended = with_pre_process if parameter == 'pre_process' else with_post_process

    @wraps(extended)
    def decorated(*args, **kwargs):
        return extended(*args, **kwargs)

    wrapped = implementation.wrapper(decorated)
    assert wrapped(None, None, None, None, None, **{parameter: True}) == 'extended'
    assert calls == [True]
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([3, 6]))
    with pytest.raises(NotImplementedError, match='require separate adaptation'):
        wrapped(None, None, None, None, None, packed_seq_params=packed)
    assert len(calls) == 1
