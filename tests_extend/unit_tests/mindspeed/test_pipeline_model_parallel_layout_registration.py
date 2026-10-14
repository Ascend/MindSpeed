# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import importlib.util
import sys
from argparse import Namespace
from dataclasses import make_dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _source_root():
    spec = importlib.util.find_spec("mindspeed")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).resolve().parent
    return Path(__file__).resolve().parents[3] / "mindspeed"


@pytest.fixture
def layout_modules(monkeypatch):
    """Load the real feature and layout with core 0.18 metadata and no NPU runtime."""
    source_root = _source_root()
    enums = ModuleType("megatron.core.transformer.enums")
    native_layer_type = Enum("LayerType", "embedding loss encoder decoder mtp")
    enums.LayerType = native_layer_type
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(get_virtual_pipeline_model_parallel_world_size=lambda: None)
    transformer = ModuleType("megatron.core.transformer")
    transformer.enums = enums
    feature_base = ModuleType("mindspeed.features_manager.feature")
    feature_base.MindSpeedFeature = object
    patch_utils = ModuleType("mindspeed.patch_utils")
    patch_utils.MindSpeedPatchesManager = Mock
    transformer_config = ModuleType("megatron.core.transformer.transformer_config")
    transformer_config.TransformerConfig = make_dataclass(
        "TransformerConfig", ["pipeline_model_parallel_layout"]
    )
    for module in (enums, core, transformer, feature_base, patch_utils, transformer_config):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, source_root / relative_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    prefix = "mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout"
    layout = load(f"{prefix}.layout", "core/pipeline_parallel/pipeline_model_parallel_layout/layout.py")
    load(f"{prefix}.adaptor", "core/pipeline_parallel/pipeline_model_parallel_layout/adaptor.py")
    feature_module = load(
        "mindspeed.features_manager.pipeline_parallel.pipeline_model_parallel_layout_feature",
        "features_manager/pipeline_parallel/pipeline_model_parallel_layout_feature.py",
    )
    feature = object.__new__(feature_module.PipelineModelParallelLayoutFeature)
    feature.feature_name = "pipeline_model_parallel_layout"
    return feature, layout, native_layer_type


@pytest.mark.parametrize("as_list", [False, True], ids=["string-layout", "enum-layout"])
def test_register_layout_patches_preserves_layer_type(layout_modules, as_list):
    feature, layout_module, native_layer_type = layout_modules
    layer_type = native_layer_type
    layout_input = (
        [[layer_type.embedding, layer_type.decoder], [layer_type.decoder, layer_type.loss]]
        if as_list else "Et|tL"
    )
    manager = Mock()
    feature.register_patches(manager, Namespace(pipeline_model_parallel_layout=layout_input))
    patches = dict(call.args for call in manager.register_patch.call_args_list)
    assert "megatron.core.transformer.enums.LayerType" not in patches
    assert layout_module.LayerType is layer_type

    layout = layout_module.PipelineParallelLayerLayout(layout_input, pipeline_model_parallel_size=2)
    assert layout.validate_layer_layout(num_layers=2, mtp_num_layers=None) is False
    assert layout.get_num_layers_to_build(layer_type.decoder, pp_rank=0) == 1
    assert layout.get_layer_offset(layer_type.decoder, pp_rank=1) == 1
    config = SimpleNamespace(pipeline_model_parallel_layout=layout)
    original = Mock(side_effect=AssertionError("custom layout should handle the call"))
    count = patches["megatron.core.transformer.transformer_block.get_num_layers_to_build"](original)
    offset = patches["megatron.core.transformer.transformer_layer.get_transformer_layer_offset"](original)
    assert count(config, pp_rank=0) == 1
    assert offset(config, pp_rank=1) == 1
    original.assert_not_called()


@pytest.mark.parametrize("layout_input", [None, ""])
def test_disabled_layout_does_not_register_patches(layout_modules, layout_input):
    feature, _, _ = layout_modules
    manager = Mock()
    feature.register_patches(manager, Namespace(pipeline_model_parallel_layout=layout_input))
    manager.register_patch.assert_not_called()


def _layout_args(layout_input):
    return Namespace(
        pipeline_model_parallel_layout=layout_input,
        pipeline_model_parallel_size=4,
        num_layers_per_virtual_pipeline_stage=None,
        num_virtual_stages_per_pipeline_rank=None,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        overlap_p2p_comm=True,
        align_param_gather=True,
    )


@pytest.mark.parametrize(
    "layout_input, expected_vpp_size",
    [("Ett|(tttt|)*6ttttL", 2), ("Et|t|t|tL", None)],
)
def test_layout_pre_validation_preserves_native_args(layout_modules, layout_input, expected_vpp_size):
    feature, layout_module, _ = layout_modules
    args = _layout_args(layout_input)
    original_args = vars(args).copy()

    feature.pre_validate_args(args)
    # MindSpeed-LLM and MegatronAdaptor can both run feature pre-validation.
    feature.pre_validate_args(args)
    assert vars(args) == original_args
    # Native Megatron validation derives VPP from the layout without injecting
    # the mutually exclusive num_virtual_stages_per_pipeline_rank option.
    num_stages = layout_module.PipelineParallelLayerLayout.get_num_stages_from_str(layout_input)
    detected_vpp_size = num_stages // args.pipeline_model_parallel_size
    args.virtual_pipeline_model_parallel_size = detected_vpp_size if detected_vpp_size > 1 else None

    feature.post_validate_args(args)
    feature.validate_args(args)
    assert args.virtual_pipeline_model_parallel_size == expected_vpp_size
    assert args.num_virtual_stages_per_pipeline_rank is None
    assert args.decoder_first_pipeline_num_layers is None


@pytest.mark.parametrize("option", ["num_layers_per_virtual_pipeline_stage", "num_virtual_stages_per_pipeline_rank"])
def test_layout_validation_rejects_explicit_conflicting_options(layout_modules, option):
    feature, _, _ = layout_modules
    args = _layout_args("Ett|(tttt|)*6ttttL")
    setattr(args, option, 2)

    feature.pre_validate_args(args)
    # Leave explicit conflicts visible to native Megatron validation.
    assert getattr(args, option) == 2
    feature.post_validate_args(args)
    with pytest.raises(AssertionError, match="No more than one"):
        feature.validate_args(args)
