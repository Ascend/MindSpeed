"""This module aims to make adaptor for megatron.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from functools import wraps

from megatron.core.parallel_state import get_nccl_options

from mindspeed.core.pipeline_parallel import flexible_schedules
from .parallel_state import (
    initialize_model_parallel_impl,
    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM,
    get_pipeline_parallel_group_for_new_stream,
)


def mindspeed_get_forward_backward_func():
    """Get forward and backward function for multi parameter model.

    Returns:
        Callable: A fun that run interleaved 1F1B schedule
        (model split into model chunks), with communication between
        pipeline stages as needed for multi parameter.
    """
    flexible_schedules.get_pipeline_parallel_group_for_new_stream = (
        get_pipeline_parallel_group_for_new_stream
    )
    return flexible_schedules.forward_backward_pipelining_without_interleaving


def mindspeed_initialize_model_parallel_wrapper(fn):
    """Wrap the initialize_model_parallel function of megatron."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        initialize_model_parallel_impl(
            *args,
            **kwargs,
            get_nccl_options=get_nccl_options,
        )  # megatron api has no `get_nccl_options` param

    return wrapper


def mindspeed_destroy_model_parallel_wrapper(destroy_model_parallel):
    """Wrap the destroy_model_parallel function of megaron."""

    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None

    return wrapper
