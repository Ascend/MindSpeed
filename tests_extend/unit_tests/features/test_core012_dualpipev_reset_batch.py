"""CPU regressions; run with pytest --noconftest.

Execute the real batch/registration function bodies with mocked device transfers
and TP collectives. This does not exercise NPU kernels or the pipeline scheduler.
"""

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import mindspeed

# CI copies tests into Megatron; resolve production sources from the package.
ROOT = Path(mindspeed.__file__).resolve().parent.parent
DUAL_FEATURE = ROOT / 'mindspeed/features_manager/pipeline_parallel/dualpipev_feature.py'
RESET_FEATURE = ROOT / 'mindspeed/features_manager/transformer/flash_attention/reset_attention_mask_feature.py'
BATCH_SOURCE = ROOT / 'mindspeed/core/context_parallel/get_batch_utils.py'
DUAL_SOURCE = ROOT / 'mindspeed/core/pipeline_parallel/dualpipev/dualpipev_chunks.py'
TARGET = 'megatron.training.utils.get_batch_on_this_tp_rank'


def function_node(path, name, parent=None):
    body = ast.parse(path.read_text(encoding='utf-8')).body
    if parent:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == parent)
    return next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)


def load_function(node, namespace):
    unit = ast.Module(body=[node], type_ignores=[])
    exec(compile(unit, '<source-extracted MindSpeed function>', 'exec'), namespace)  # noqa: S102  # pylint: disable=exec-used
    return namespace[node.name]


def unused_function(name):
    def unused(*args, **kwargs):
        raise AssertionError('Registration must not execute imported functions')

    unused.__name__ = name
    return unused


@pytest.mark.parametrize('dual,reset', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('reverse_order', [False, True])
def test_batch_patch_registration(monkeypatch, dual, reset, reverse_order):
    manager = runpy.run_path(str(ROOT / 'mindspeed/patch_utils.py'))['MindSpeedPatchesManager']
    methods = [
        function_node(DUAL_FEATURE, 'register_patches', 'DualpipeVFeature'),
        function_node(RESET_FEATURE, 'register_patches', 'ResetAttentionMaskFeature'),
    ]
    modules = {}
    for method in methods:
        for node in ast.walk(method):
            if isinstance(node, ast.ImportFrom):
                module = modules.setdefault(node.module, ModuleType(node.module))
                for alias in node.names:
                    setattr(module, alias.name, unused_function(alias.name))
    # Keep actual implementations as the registration candidates.
    dual_batch = load_function(function_node(DUAL_SOURCE, 'get_batch_on_this_tp_rank'), {})
    reset_batch = load_function(function_node(BATCH_SOURCE, 'get_batch_on_this_tp_rank'), {})
    modules['mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks'].get_batch_on_this_tp_rank = dual_batch
    modules['mindspeed.core.context_parallel.get_batch_utils'].get_batch_on_this_tp_rank = reset_batch
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    args = SimpleNamespace(
        schedules_method='dualpipev' if dual else None,
        reset_attention_mask=reset,
        mtp_num_layers=1,
        multi_latent_attention=False,
    )
    for method in reversed(methods) if reverse_order else methods:
        load_function(method, {})(SimpleNamespace(feature_name='reset_attention_mask'), manager, args)
    if reset or dual:
        assert manager.patches_info[TARGET].patch_func is (reset_batch if reset else dual_batch)
    else:
        assert TARGET not in manager.patches_info
    # Selecting the reset batch must not remove DualPipeV's other patches.
    assert ('megatron.training.training.train_step' in manager.patches_info) == dual


@pytest.mark.parametrize(
    'dual,stage,pp_size',
    [
        (False, 'first', 1),
        (False, 'first', 4),
        (False, 'middle', 4),
        (False, 'last', 4),
        (True, 'first', 4),
        (True, 'middle', 4),
        (True, 'last', 4),
    ],
)
@pytest.mark.parametrize('mtp', [None, 1])
@pytest.mark.parametrize('attention_mask', [False, True])
def test_tp_batch_and_boundaries(monkeypatch, dual, stage, pp_size, mtp, attention_mask):
    args = SimpleNamespace(
        schedules_method='dualpipev' if dual else None,
        pipeline_model_parallel_size=pp_size,
        mtp_num_layers=mtp,
        reset_attention_mask=True,
        micro_batch_size=1,
        seq_length=6,
        create_attention_mask_in_dataloader=attention_mask,
        attention_mask_type='causal',
        context_parallel_size=1,
    )
    data = {
        'tokens': torch.arange(6).reshape(1, 6),
        'labels': torch.arange(1, 7).reshape(1, 6),
        'loss_mask': torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]]),
        'position_ids': torch.tensor([[0, 1, 2, 0, 1, 2]]),
        'actual_seq_len': torch.tensor([3, 6]),
    }
    if attention_mask:
        data['attention_mask'] = torch.zeros((1, 1, 6, 6), dtype=torch.bool)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    state = SimpleNamespace(rank=0)
    sent, received, boundaries = [], [], []
    queue = []

    def broadcast(item):
        if item is None:
            return
        if state.rank == 0:
            name = next(key for key, value in data.items() if value is item)
            sent.append(name)
            queue.append((name, item.clone()))
        else:
            name, value = queue.pop(0)
            assert item.shape == value.shape and item.dtype == value.dtype
            item.copy_(value)
            received.append(name)

    def broadcast_dynamic(item):
        if state.rank == 0:
            assert item is data['actual_seq_len']
            queue.append(('actual_seq_len', item.clone()))
            return item
        assert item is None
        name, value = queue.pop(0)
        assert name == 'actual_seq_len'
        return value

    namespace = {
        'torch': torch,
        'get_args': lambda: args,
        'mpu': SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: state.rank,
            is_pipeline_first_stage=lambda: stage == 'first',
            is_pipeline_last_stage=lambda: stage == 'last',
        ),
        '_broadcast': broadcast,
        'broadcast_dynamic': broadcast_dynamic,
        'set_actual_seq_len': lambda value: boundaries.append(value.clone()),
    }
    get_batch = load_function(function_node(BATCH_SOURCE, 'get_batch_on_this_tp_rank'), namespace)
    iterator = iter([data])
    source = get_batch(iterator)
    assert next(iterator, None) is None
    state.rank = 1
    receiver = get_batch(None)
    assert not queue
    assert sent == received
    full = pp_size == 1 or (dual and (stage == 'first' or (stage == 'last' and mtp is not None)))
    if full:
        expected = ['tokens', 'labels', 'loss_mask', 'attention_mask', 'position_ids']
    elif stage == 'first':
        expected = ['tokens', 'attention_mask', 'position_ids']
        assert receiver['labels'] is None and receiver['loss_mask'] is None
    elif stage == 'last':
        expected = ['labels', 'loss_mask', 'attention_mask', 'position_ids']
        assert receiver['tokens'] is None
    else:
        expected = ['position_ids']
    if not attention_mask and 'attention_mask' in expected:
        expected.remove('attention_mask')
    assert sent == expected
    for key in expected:
        assert torch.equal(source[key], data[key])
        assert torch.equal(receiver[key], data[key])
    assert len(boundaries) == 2
    assert all(torch.equal(value, data['actual_seq_len']) for value in boundaries)
    assert args.pipeline_model_parallel_size == pp_size
