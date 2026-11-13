# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd.
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

from functools import wraps
from logging import getLogger

from megatron.core.parallel_state import RankGenerator
from megatron.core.parallel_state import get_nccl_options, create_group
from megatron.training.utils import get_args

from mindspeed.core.transformer.moe.moe_feature.balanced_moe.modules.moe_layer import BalancedMoELayer
from mindspeed.core.transformer.moe.moe_feature.balanced_moe.parallel_state import (
    initialize_model_parallel_impl,
)
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.experts import MindSpeedFbOverlapGmmExperts
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.shared_experts import SharedExpertMLPFbOverlap


def mindspeed_initialize_model_parallel_wrapper(fn):
    """Wrap the initialize_model_parallel function of megatron."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        initialize_model_parallel_impl(
            *args,
            **kwargs,
            create_group=create_group,
            get_nccl_options=get_nccl_options,
            rank_generator=RankGenerator,
            get_args=get_args,
        )  # megatron api has no `get_nccl_options` param

    return wrapper


def get_moe_module_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        spec = fn(*args, **kwargs)
        args = get_args()
        if getattr(args, 'balanced_moe_experts', False):
            log = getLogger(__name__)
            log.info("balanced_moe_experts is enabled. Replacing default megatron layer spec for moe...")
            spec.module = BalancedMoELayer
            spec.submodules.experts.module = MindSpeedFbOverlapGmmExperts
            spec.submodules.shared_experts.module = SharedExpertMLPFbOverlap

        return spec

    return wrapper
