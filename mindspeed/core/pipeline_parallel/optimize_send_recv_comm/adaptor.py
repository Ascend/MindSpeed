"""This module aims to make adaptor for megatron.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from mindspeed.core.pipeline_parallel import flexible_schedules


def mindspeed_get_forward_backward_func():
    """Get forward and backward function for multi parameter model.

    Returns:
        Callable: A fun that run interleaved 1F1B schedule
        (model split into model chunks), with communication between
        pipeline stages as needed for multi parameter.
    """
    return flexible_schedules.forward_backward_pipelining_without_interleaving
