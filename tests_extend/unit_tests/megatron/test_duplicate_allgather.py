# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU checks for parameter-sync ownership; no distributed/NPU runtime needed."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _mindspeed_source_root():
    """Find MindSpeed even when tests_extend is copied into Megatron-LM."""
    # Resolve the active package without importing its training/NPU modules.
    try:
        package_spec = importlib.util.find_spec('mindspeed')
    except (ImportError, ValueError):
        package_spec = None

    candidates = []
    if package_spec is not None:
        for package_dir in package_spec.submodule_search_locations or ():
            candidates.append(Path(package_dir).resolve().parent)
    # Also support running directly from an uninstalled MindSpeed checkout.
    candidates.append(Path(__file__).resolve().parents[3])
    for root in candidates:
        if (root / 'mindspeed/core/optimizer/fix_duplicate_allgather.py').is_file():
            return root

    raise FileNotFoundError(
        "Cannot locate MindSpeed sources. Install the matching checkout with "
        "'python -m pip install -e /path/to/MindSpeed' or add it to PYTHONPATH. "
        "Copied tests_extend files alone do not contain the production code."
    )


ROOT = _mindspeed_source_root()


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name='sync_patch')
def fixture_sync_patch(monkeypatch):
    module_name = 'megatron.core.optimizer.layer_wise_optimizer'
    layer_wise = ModuleType(module_name)
    layer_wise._bucket_is_managed_by_layer_wise_optimizer = lambda bucket, default_for_untagged: getattr(
        bucket, 'layer_wise', default_for_untagged
    )
    monkeypatch.setitem(sys.modules, module_name, layer_wise)
    return load_source('duplicate_allgather_patch', 'mindspeed/core/optimizer/fix_duplicate_allgather.py')


def make_group(*buckets):
    return SimpleNamespace(buckets=list(buckets), start_param_sync=Mock(), post_param_sync=Mock())


class Bucket:
    """Use the same object identity semantics as MCore's _ParamAndGradBucket."""

    def __init__(self, layer_wise=False):
        self.layer_wise = layer_wise


class ModelChunk:
    def __init__(self, dense_groups, expert_groups):
        self.bucket_groups = dense_groups
        self.expert_parallel_bucket_groups = expert_groups

    def _start_bucket_group_param_sync(self, group, force_sync):
        # Match the synchronous DDP dispatch contract: communication plus post-processing.
        group.start_param_sync(force_sync=force_sync)
        group.post_param_sync()


def optimizer(chunks, buckets):
    return SimpleNamespace(
        is_stub_optimizer=False,
        model_chunks=chunks,
        buffers=[SimpleNamespace(buckets=buckets)],
    )


@pytest.mark.parametrize('num_chunks', [1, 2])
@pytest.mark.parametrize('has_experts', [False, True])
def test_each_optimizer_syncs_only_its_own_updated_parameters(sync_patch, num_chunks, has_experts):
    chunks, dense_buckets, expert_buckets = [], [], []
    for _ in range(num_chunks):
        # A group can aggregate several dtype buffers; use the original DDP group.
        dense = [Bucket(), Bucket()]
        expert = [Bucket()] if has_experts else []
        chunks.append(ModelChunk([make_group(*dense)], [make_group(*expert)] if expert else []))
        dense_buckets.extend(dense)
        expert_buckets.extend(expert)

    dense_opt = optimizer(chunks, dense_buckets)
    expert_opt = optimizer(chunks, expert_buckets)
    for step in range(1, 3):
        sync_patch.start_param_sync_for_bucket_group_subset(dense_opt)
        for chunk in chunks:
            for group in chunk.bucket_groups:
                assert group.start_param_sync.call_count == step
                assert group.post_param_sync.call_count == step
                group.start_param_sync.assert_called_with(force_sync=False)
            for group in chunk.expert_parallel_bucket_groups:
                # Expert sync must not happen before the expert optimizer runs.
                assert group.start_param_sync.call_count == step - 1

        sync_patch.start_param_sync_for_bucket_group_subset(expert_opt)
        for chunk in chunks:
            for group in chunk.bucket_groups + chunk.expert_parallel_bucket_groups:
                assert group.start_param_sync.call_count == step
                assert group.post_param_sync.call_count == step


def test_skips_empty_and_layer_wise_groups(sync_patch):
    bucket = Bucket(layer_wise=True)
    empty, layer_wise = make_group(), make_group(bucket)
    opt = optimizer([ModelChunk([empty, layer_wise], [])], [bucket])
    sync_patch.start_param_sync_for_bucket_group_subset(opt)
    empty.start_param_sync.assert_not_called()
    layer_wise.start_param_sync.assert_not_called()


def test_stub_optimizer_never_syncs_other_optimizers_parameters(sync_patch):
    # Stub optimizers return from __init__ before buffers are initialized.
    sync_patch.start_param_sync_for_bucket_group_subset(SimpleNamespace(is_stub_optimizer=True))


def test_rejects_group_with_partial_ownership(sync_patch):
    owned, foreign = Bucket(), Bucket()
    group = make_group(owned, foreign)
    opt = optimizer([ModelChunk([group], [])], [owned])
    with pytest.raises(RuntimeError, match='spans multiple distributed optimizers'):
        sync_patch.start_param_sync_for_bucket_group_subset(opt)
    group.start_param_sync.assert_not_called()


def test_npu_enhancement_registers_new_optimizer_entry_by_default(sync_patch, monkeypatch):
    modules = {
        'mindspeed.features_manager.feature': {'MindSpeedFeature': object},
        'mindspeed.core.megatron_basic.count_zero_fix': {'step': Mock()},
        'mindspeed.core.megatron_basic.megatron_basic': {'dist_optim_load_state_dict': Mock()},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'mindspeed.core.optimizer.fix_duplicate_allgather', sync_patch)
    feature = load_source(
        'npu_enhancement_registration', 'mindspeed/features_manager/npu_enhancement/npu_enhancement.py'
    )
    parser = argparse.ArgumentParser()
    feature.NpuEnhancementFeature.register_args(SimpleNamespace(add_parser_argument_choices_value=Mock()), parser)
    args = parser.parse_args([])
    assert not hasattr(args, 'fix_duplicate_param_all_gather')
    manager = Mock()
    feature.NpuEnhancementFeature._register_bugfix_patches(None, manager)
    registered = {call.args[0]: call.args[1] for call in manager.register_patch.call_args_list}
    target = 'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.start_param_sync_for_bucket_group_subset'
    assert registered[target] is sync_patch.start_param_sync_for_bucket_group_subset
    # The pre-existing DDP patch remains registered.
    ddp_target = 'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_param_sync'
    assert registered[ddp_target] is sync_patch.start_param_sync
