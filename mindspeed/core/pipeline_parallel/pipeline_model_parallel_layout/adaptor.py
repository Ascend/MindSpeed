# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps

from .layout import LayerType, PipelineParallelLayerLayout


def _get_layout(config):
    return getattr(config, "pipeline_model_parallel_layout", None)


def get_num_layers_to_build_wrapper(fn):
    """Use custom pipeline layout to determine decoder layers in the current stage."""

    @wraps(fn)
    def wrapper(config, *args, **kwargs):
        layout = _get_layout(config)
        if layout is not None:
            return layout.get_num_layers_to_build(layer_type=LayerType.decoder)
        return fn(config, *args, **kwargs)

    return wrapper


def get_transformer_layer_offset_wrapper(fn):
    """Use custom pipeline layout to determine decoder layer offset."""

    @wraps(fn)
    def wrapper(config, *args, **kwargs):
        layout = _get_layout(config)
        if layout is not None:
            return layout.get_layer_offset(layer_type=LayerType.decoder)
        return fn(config, *args, **kwargs)

    return wrapper


def apply_pipeline_model_parallel_layout_to_config(config):
    """Parse and validate custom pipeline layout on TransformerConfig."""
    if not hasattr(config, "pipeline_model_parallel_layout"):
        from mindspeed.args_utils import get_full_args

        args = get_full_args()
        config.pipeline_model_parallel_layout = getattr(args, "pipeline_model_parallel_layout", None)
    if config.pipeline_model_parallel_layout is None:
        return

    # If pipeline layout is set, check conflicts with other pipeline layout arguments.
    any_conflict = (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
        or config.account_for_embedding_in_pipeline_split
        or config.account_for_loss_in_pipeline_split
    )
    if any_conflict:
        raise ValueError(
            "pipeline_model_parallel_layout cannot be set"
            " with other pipeline layout arguments."
            f" {config.num_layers_in_first_pipeline_stage=},"
            f" {config.num_layers_in_last_pipeline_stage=},"
            f" {config.account_for_embedding_in_pipeline_split=},"
            f" {config.account_for_loss_in_pipeline_split=}."
        )

    # Transfer pipeline_model_parallel_layout from str or list to PipelineParallelLayerLayout.
    if isinstance(config.pipeline_model_parallel_layout, str):
        config.pipeline_model_parallel_layout = PipelineParallelLayerLayout.from_str(
            layout=config.pipeline_model_parallel_layout,
            pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        )
    elif isinstance(config.pipeline_model_parallel_layout, list):
        # Since list is not hashable, the initialization will not be cached.
        config.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
            layout=config.pipeline_model_parallel_layout,
            pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        )
    elif not isinstance(config.pipeline_model_parallel_layout, PipelineParallelLayerLayout):
        raise TypeError(
            "pipeline_model_parallel_layout must be a str, list, or "
            f"PipelineParallelLayerLayout, but got {type(config.pipeline_model_parallel_layout)}"
        )

    # Check whether the input VPP size conflicts with the PP layout.
    detected_vpp_size = config.pipeline_model_parallel_layout.virtual_pipeline_model_parallel_size
    if config.virtual_pipeline_model_parallel_size is not None:
        assert config.virtual_pipeline_model_parallel_size == detected_vpp_size, (
            f"virtual_pipeline_model_parallel_size conflicts with"
            f" pipeline_model_parallel_layout,"
            f" ({config.virtual_pipeline_model_parallel_size=}, "
            f" {detected_vpp_size=})"
        )
    elif detected_vpp_size > 1:
        config.virtual_pipeline_model_parallel_size = detected_vpp_size

    # Check whether the layout is valid.
    config.mtp_standalone = config.pipeline_model_parallel_layout.validate_layer_layout(
        num_layers=config.num_layers, mtp_num_layers=getattr(config, "mtp_num_layers", None)
    )


def transformer_config_post_init_wrapper(fn):
    """Decorate TransformerConfig.__post_init__ with custom pipeline layout validation."""

    @wraps(fn)
    def wrapper(self):
        fn(self)
        apply_pipeline_model_parallel_layout_to_config(self)

    return wrapper
