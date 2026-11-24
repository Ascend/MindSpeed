# Copyright 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

from typing import Any, List, Optional, Tuple, TypedDict


class ParamGroupOverride(TypedDict, total=False):
    """Override values for a parameter group. These values may be optimizer-state/scheduler related.

    These are the values you see later in param_group.get(...) calls in the
        OptimizerParamScheduler.get_lr and get_wd methods. If you use a custom optimizer
        or scheduler, you could override those variables instead.

    Example:
        >>> param_group_override = ParamGroupOverride(min_lr=1e-4, wd_mult=0.1)
        >>> param_group_override == ParamGroupOverride(optimizer='muon')  # per-param optimizer

    """

    max_lr: float
    min_lr: float
    start_wd: float
    end_wd: float
    wd_mult: float
    optimizer: str


def param_group_override_to_tuple(
    param_group_override: Optional[ParamGroupOverride],
) -> Optional[Tuple[Tuple[str, Any], ...]]:
    """Convert a param group override to a tuple for use as a key in a dictionary.

    The tuple is sorted by the keys of the param group override to handle different orderings of
     the keys in different override dictionaries which still mean the same thing.
    """
    if param_group_override is None:
        return None
    return tuple(sorted(param_group_override.items()))


def combine_param_group_overrides(
    param_group_overrides: List[Optional[ParamGroupOverride]],
) -> ParamGroupOverride:
    """Combine a list of param group overrides into a single param group override.

    This function ensures that the overrides are not conflicting as well.

    Args:
        param_group_overrides (list[ParamGroupOverride]): list of param group overrides to combine

    Returns:
        ParamGroupOverride: combined param group override
    """
    combined_override = ParamGroupOverride()
    for override in param_group_overrides:
        if override is None:
            continue
        for key, value in override.items():
            if key in combined_override:
                if combined_override[key] != value:
                    raise ValueError(f"Conflicting overrides for {key}: {combined_override[key]} and {value}")
            combined_override[key] = value
    return combined_override
