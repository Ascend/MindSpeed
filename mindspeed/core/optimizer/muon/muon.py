# Copyright 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pylint: skip-file
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

from typing import Any


def get_megatron_muon_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible Muon optimizer getter."""
    from megatron.core.optimizer import get_megatron_optimizer

    use_layer_wise = kwargs.pop("layer_wise_distributed_optimizer", False)

    if "config" in kwargs:
        config = kwargs["config"]
    else:
        config = args[0]

    if use_layer_wise and not getattr(config, "optimizer", "").startswith("dist_"):
        raise ValueError("Layer-wise distributed optimizer is enabled by dist_ prefix in optimizer name.")

    return get_megatron_optimizer(*args, **kwargs)
