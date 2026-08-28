# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace

import pytest

from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.adaptor import (
    apply_pipeline_model_parallel_layout_to_config,
)
from mindspeed.core.pipeline_parallel.pipeline_model_parallel_layout.layout import (
    LayerType,
    PipelineParallelLayerLayout,
    parallel_state,
)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("empty_stage_and_multiplication", id="empty-stage-and-multiplication"),
        pytest.param("vpp_stage_mapping", id="vpp-stage-mapping"),
        pytest.param("mtp_standalone", id="mtp-standalone"),
        pytest.param("invalid_character", id="invalid-character"),
        pytest.param("stage_count_not_divisible", id="stage-count-not-divisible-by-pp"),
        pytest.param("missing_embedding", id="missing-embedding"),
        pytest.param("missing_loss", id="missing-loss"),
        pytest.param("decoder_count_mismatch", id="decoder-count-mismatch"),
        pytest.param("mtp_count_mismatch", id="mtp-count-mismatch"),
        pytest.param("conflicting_config", id="conflicting-layout-config"),
    ],
)
def test_pipeline_model_parallel_layout(case, monkeypatch):
    """Validate parsing, stage mapping, MTP handling, and rejected layouts."""
    if case == "empty_stage_and_multiplication":
        layout = PipelineParallelLayerLayout("E||t*2|L", pipeline_model_parallel_size=4)
        assert layout.layout == [
            [[LayerType.embedding]],
            [[]],
            [[LayerType.decoder, LayerType.decoder]],
            [[LayerType.loss]],
        ]
        assert layout.validate_layer_layout(num_layers=2, mtp_num_layers=None) is False
        assert layout.get_num_layers_to_build(pp_rank=1) == 0
        assert layout.get_layer_id_list(pp_rank=2) == [0, 1]
        return

    if case == "vpp_stage_mapping":
        layout = PipelineParallelLayerLayout("Et|t|t|mL", pipeline_model_parallel_size=2)
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_rank", lambda: 1)

        assert layout.virtual_pipeline_model_parallel_size == 2
        assert layout.validate_layer_layout(num_layers=3, mtp_num_layers=1) is False
        assert layout.get_layer_id_list(pp_rank=0, vp_stage=0) == [0]
        assert layout.get_layer_id_list(pp_rank=1, vp_stage=0) == [1]
        assert layout.get_layer_id_list(pp_rank=0, vp_stage=1) == [2]
        assert not layout.get_layer_id_list(pp_rank=1, vp_stage=1)
        assert layout.get_num_layers_to_build() == 1
        assert layout.get_layer_offset() == 2
        return

    if case == "mtp_standalone":
        layout = PipelineParallelLayerLayout("Et|m|L", pipeline_model_parallel_size=3)
        assert layout.validate_layer_layout(num_layers=1, mtp_num_layers=1) is True
        assert layout.get_num_layers_to_build(LayerType.mtp, pp_rank=1) == 1
        return

    if case == "invalid_character":
        with pytest.raises(AssertionError, match="Invalid layer character"):
            PipelineParallelLayerLayout("Ex|L", pipeline_model_parallel_size=2)
        return

    if case == "stage_count_not_divisible":
        with pytest.raises(AssertionError, match="must be divisible"):
            PipelineParallelLayerLayout("E|t|L", pipeline_model_parallel_size=2)
        return

    invalid_layouts = {
        "missing_embedding": ("t|L", 1, None, "The first layer must be embedding"),
        "missing_loss": ("E|t", 1, None, "The last layer must be loss"),
        "decoder_count_mismatch": ("Et|L", 2, None, "Number of decoder layers"),
        "mtp_count_mismatch": ("Et|mL", 1, 2, "Number of mtp layers"),
    }
    if case in invalid_layouts:
        layout_str, num_layers, mtp_num_layers, error = invalid_layouts[case]
        layout = PipelineParallelLayerLayout(layout_str, pipeline_model_parallel_size=2)
        with pytest.raises(AssertionError, match=error):
            layout.validate_layer_layout(num_layers=num_layers, mtp_num_layers=mtp_num_layers)
        return

    config = SimpleNamespace(
        pipeline_model_parallel_layout="Et|L",
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        num_layers=1,
        mtp_num_layers=None,
        num_layers_in_first_pipeline_stage=1,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )
    with pytest.raises(ValueError, match="cannot be set with other pipeline layout arguments"):
        apply_pipeline_model_parallel_layout_to_config(config)
