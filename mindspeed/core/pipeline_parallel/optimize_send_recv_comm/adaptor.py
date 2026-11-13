# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
