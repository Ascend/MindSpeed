# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU source-level checks for the Megatron 0.18 DualPipeV MTP adapter.

Run with pytest --noconftest. Real model gates, MTP forwards and loss code are
loaded without importing the distributed/NPU stack. Transformer computation
and process groups are substitutes; this is not a distributed training test.
"""

import ast
import copy
import importlib.util
import sys
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


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
MTP_UTILS = ROOT / 'mindspeed/core/pipeline_parallel/dualpipev/mtp_utils.py'
FB_LAYER = ROOT / 'mindspeed/core/transformer/moe/moe_feature/fb_overlap/transformer_layer.py'


def _node(path, name):
    node = ast.parse(path.read_text(encoding='utf-8'))
    for part in name.split('.'):
        node = next(child for child in node.body if getattr(child, 'name', None) == part)
    return copy.deepcopy(node)


def _load(path, name, namespace):
    node = _node(path, name)
    tree = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)  # noqa: S102 - selected repository code
    return namespace[node.name]


def _placement_namespace(rank, schedule='dualpipev'):
    return {
        'wraps': wraps,
        'get_args': lambda: SimpleNamespace(schedules_method=schedule),
        'mpu': SimpleNamespace(is_pipeline_first_stage=lambda **kwargs: rank == 0),
    }


@pytest.mark.parametrize('rank', [0, 1, 3])
@pytest.mark.parametrize('layers', [None, 0, 1, 2])
@pytest.mark.parametrize('ignore_virtual', [False, True])
def test_mtp_rank_follows_dualpipev_return_path(rank, layers, ignore_virtual):
    namespace = _placement_namespace(rank)
    original = Mock(side_effect=AssertionError('DualPipeV must not use the physical-last-stage predicate'))
    wrapper = _load(MTP_UTILS, 'dualpipev_mtp_on_this_rank_wrapper', namespace)(original)
    assert wrapper(mtp_num_layers=layers, ignore_virtual=ignore_virtual) == (bool(layers) and rank == 0)
    original.assert_not_called()


@pytest.mark.parametrize('schedule', [None, '1f1b'])
def test_other_schedules_keep_native_layout_and_virtual_stage_semantics(schedule):
    original = Mock(return_value=object())
    namespace = _placement_namespace(0, schedule)
    wrapper = _load(MTP_UTILS, 'dualpipev_mtp_on_this_rank_wrapper', namespace)(original)
    layout = object()
    assert wrapper(layout, 2, False, 3) is original.return_value
    original.assert_called_once_with(layout=layout, mtp_num_layers=2, ignore_virtual=False, vp_stage=3)


@pytest.fixture
def native_root():
    candidates = [ROOT.parent / 'Megatron-LM-core_r0.18.0/Megatron-LM-core_r0.18.0', ROOT.parent / 'Megatron-LM']
    spec = importlib.util.find_spec('megatron')
    if spec is not None:
        candidates.extend(Path(path).parent for path in spec.submodule_search_locations or [])
    for candidate in candidates:
        if (candidate / 'megatron/core/transformer/multi_token_prediction.py').is_file():
            return candidate
    pytest.skip('requires the matching Megatron 0.18 source tree')


@pytest.fixture
def runtime(native_root):
    mtp_source = native_root / 'megatron/core/transformer/multi_token_prediction.py'
    gpt_source = native_root / 'megatron/core/models/gpt/gpt_model.py'
    namespace = _placement_namespace(0)
    namespace.update(
        torch=torch,
        parallel_state=SimpleNamespace(
            get_pipeline_model_parallel_rank=lambda: 0,
            is_pipeline_last_stage=lambda **kwargs: False,
            get_data_parallel_group=lambda **kwargs: None,
        ),
        InferenceMode=SimpleNamespace(is_active=lambda: False),
        has_config_logger_enabled=lambda config: False,
        MTPLossLoggingHelper=SimpleNamespace(save_loss_to_tracker=lambda *args, **kwargs: None),
        get_mtp_layer_offset=lambda config, vp_stage: 0,
        nullcontext=nullcontext,
        make_viewless_tensor=lambda inp, **kwargs: inp,
        all_gather_last_dim_from_tensor_parallel_region=lambda tensor: tensor,
        MTPTransformerLayer=SimpleNamespace(apply=lambda layer, config, hidden, *args: (layer(hidden), None, None)),
    )
    native_rank = _load(mtp_source, 'mtp_on_this_rank', namespace)
    optimized_rank = _load(MTP_UTILS, 'dualpipev_mtp_on_this_rank_wrapper', namespace)(native_rank)
    for name in ('roll_tensor', '_roll_tensor_packed_seq', 'MTPLossAutoScaler', 'process_mtp_loss'):
        _load(mtp_source, name, namespace)
    roll_wrapper = _load(
        ROOT / 'mindspeed/core/transformer/multi_token_prediction.py', 'roll_tensor_packed_seq_wrapper', namespace
    )
    namespace['_roll_tensor_packed_seq'] = roll_wrapper(namespace['_roll_tensor_packed_seq'])

    # Execute the actual GPT initialization gate, including its spec check.
    init = _node(gpt_source, 'GPTModel.__init__')
    gate = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == 'mtp_process' for target in node.targets)
    )
    gate_fn = ast.parse('def initialize_mtp_gate(self, mtp_block_spec, vp_stage=None): pass').body[0]
    gate_fn.body = [gate]
    gate_code = compile(ast.fix_missing_locations(ast.Module(body=[gate_fn], type_ignores=[])), str(gpt_source), 'exec')
    exec(gate_code, namespace)  # noqa: S102 - original GPT initialization gate

    postprocess = _load(gpt_source, 'GPTModel._postprocess', namespace)
    block_forward = _load(mtp_source, 'MultiTokenPredictionBlock.forward', namespace)
    embeddings = _load(mtp_source, 'MultiTokenPredictionLayer._get_embeddings', namespace)
    fb_forward = _load(FB_LAYER, 'dualpipev_fb_overlap_mtp_layer_forward', namespace)
    predicted_logits = _load(
        ROOT / 'mindspeed/core/tensor_parallel/cross_entropy.py', 'calculate_predicted_logits', namespace
    )

    class Projection(torch.nn.Linear):
        def forward(self, hidden, **kwargs):
            return super().forward(hidden), None

    class Embedding(torch.nn.Embedding):
        def forward(self, input_ids, position_ids):
            return super().forward(input_ids).transpose(0, 1).contiguous()

    class Layer(torch.nn.Module):
        forward = fb_forward
        _get_embeddings = embeddings

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.cp_group = None
            self.sequence_parallel = False
            self.enorm = torch.nn.Identity()
            self.hnorm = torch.nn.Identity()
            self.final_layernorm = torch.nn.Identity()
            self.eh_proj = Projection(8, 4, bias=False)
            # Match 0.18's actual attribute; do not invent the old alias.
            self.mtp_model_layer = torch.nn.Linear(4, 4, bias=False)

    class Block(torch.nn.Module):
        forward = block_forward

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.vp_stage = None
            self.mtp_use_repeated_layer = False
            self.layers = torch.nn.ModuleList(Layer(config) for _ in range(config.mtp_num_layers))

    class Model(torch.nn.Module):
        _postprocess = postprocess

        def __init__(self, rank_predicate, layers, has_spec=True):
            super().__init__()
            self.config = SimpleNamespace(
                pipeline_model_parallel_layout=None,
                mtp_num_layers=layers,
                mtp_loss_scaling_factor=0.3,
                calculate_per_token_loss=False,
                sequence_parallel=False,
                fp8=False,
                recompute_granularity=None,
                use_mup=False,
            )
            namespace['mtp_on_this_rank'] = rank_predicate
            namespace['initialize_mtp_gate'](self, object() if has_spec else None)
            if self.mtp_process:
                self.embedding = Embedding(8, 4)
                self.mtp = Block(self.config)
            self.post_process = True
            self.share_embeddings_and_output_weights = False
            self.output_layer = Projection(4, 8, bias=False)
            self.pg_collection = SimpleNamespace(cp=None)
            self.loss_shapes = []

        def _scale_logits(self, logits):
            return logits

        def compute_language_model_loss(self, labels, logits):
            self.loss_shapes.append((logits.shape[:-1], labels.numel()))
            target = labels.transpose(0, 1).contiguous()
            # Execute the exact indexing operation from the reported failure.
            # The normal CE autograd Function runs this part under no_grad.
            with torch.no_grad():
                predicted_logits(logits.detach().clone(), target, logits.detach().max(dim=-1).values, 0, 8)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8), target.reshape(-1), reduction='none')
            return loss.reshape_as(target).transpose(0, 1).contiguous()

    return SimpleNamespace(Model=Model, native_rank=native_rank, optimized_rank=optimized_rank)


def _run_postprocess(model, sequence=16384, batch=1, packed=True):
    torch.manual_seed(79)
    ids = torch.arange(sequence * batch).reshape(batch, sequence) % 8
    labels = (ids + 1) % 8
    hidden = torch.randn(sequence, batch, 4, requires_grad=True)
    boundaries = SimpleNamespace(cu_seqlens_q=torch.tensor([0, sequence // 3, sequence])) if packed else None
    loss = model._postprocess(
        hidden_states=hidden,
        input_ids=ids,
        position_ids=torch.arange(sequence).expand(batch, -1),
        labels=labels,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=model.mtp_process,
        loss_mask=torch.ones_like(labels, dtype=torch.float32),
        packed_seq_params=boundaries,
    )
    return hidden, loss


def test_native_physical_last_rank_gate_reproduces_8192_vs_16384(runtime):
    model = runtime.Model(runtime.native_rank, layers=1)
    assert not model.mtp_process
    assert not hasattr(model, 'mtp')
    with pytest.raises((IndexError, RuntimeError), match='8192.*16384'):
        _run_postprocess(model)
    assert model.loss_shapes == [(torch.Size([8192, 1]), 16384)]


@pytest.mark.parametrize('layers,sequence,batch', [(1, 16384, 1), (1, 16, 2), (2, 16, 2)])
@pytest.mark.parametrize('packed', [False, True])
def test_mtp_forward_and_loss_keep_full_token_count_and_gradients(runtime, layers, sequence, batch, packed):
    model = runtime.Model(runtime.optimized_rank, layers)
    assert model.mtp_process
    hidden, loss = _run_postprocess(model, sequence, batch, packed)
    assert loss.shape == (batch, sequence)
    assert model.loss_shapes == [(torch.Size([sequence, batch]), sequence * batch)] * (layers + 1)
    loss.mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for layer in model.mtp.layers:
        grad = layer.mtp_model_layer.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_outbound_chunk_without_mtp_spec_stays_without_mtp(runtime):
    model = runtime.Model(runtime.optimized_rank, layers=1, has_spec=False)
    assert not model.mtp_process
    assert not hasattr(model, 'mtp')
    assert not hasattr(model, 'embedding')


@pytest.mark.parametrize('schedule,layers', [('dualpipev', 1), ('dualpipev', None), (None, 1)])
def test_feature_registers_rank_predicate_only_for_dualpipev_mtp(monkeypatch, schedule, layers):
    def module(name, exports):
        item = ModuleType(name)
        item.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, item)

    module('megatron.training.utils', {'print_rank_0': Mock()})
    module(
        'mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules',
        {'forward_backward_pipelining_with_cutinhalf': Mock()},
    )
    module(
        'mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks',
        {
            name: Mock()
            for name in (
                'get_model',
                'dualpipev_fp16forward',
                'get_num_layers_to_build',
                'train_step',
                '_allreduce_embedding_grads_wrapper',
                'dualpipev_get_batch_on_this_tp_rank_wrapper',
                'evaluate',
                'get_transformer_layer_offset',
                'pretrain',
            )
        },
    )
    wrapper = _load(MTP_UTILS, 'dualpipev_mtp_on_this_rank_wrapper', _placement_namespace(0))
    module(
        'mindspeed.core.pipeline_parallel.dualpipev.mtp_utils',
        {
            'setup_embeddings_and_output_layer_with_mtp': Mock(),
            'dualpipev_get_mtp_num_layers_to_build': Mock(),
            'dualpipev_mtp_on_this_rank_wrapper': wrapper,
        },
    )
    feature = _load(
        ROOT / 'mindspeed/features_manager/pipeline_parallel/dualpipev_feature.py',
        'DualpipeVFeature',
        {'MindSpeedFeature': object},
    )
    recorder = Mock()
    feature.register_patches(None, recorder, SimpleNamespace(schedules_method=schedule, mtp_num_layers=layers))
    target = 'megatron.core.transformer.multi_token_prediction.mtp_on_this_rank'
    matches = [call.args for call in recorder.register_patch.call_args_list if call.args[0] == target]
    assert matches == ([(target, wrapper)] if schedule == 'dualpipev' and layers else [])
