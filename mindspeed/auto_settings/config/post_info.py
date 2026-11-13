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

from dataclasses import dataclass

from mindspeed.auto_settings.config.model_config import ModelConfig


@dataclass
class PostInfo:
    FILENAME = "auto_settings_post_info.pkl"

    model_config: ModelConfig = None
    devices_per_node: int = None  # type: ignore
    nnodes: int = None  # type: ignore
    node_rank: int = None  # type: ignore
    device_type: str = None  # type: ignore
    wait_timeout: int = None  # type: ignore
    memory_cap: float = None  # type: ignore
    driver_version: str = None  # type: ignore
    cann_version: str = None  # type: ignore
    mm_model: str = None # type: ignore