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

import torch
from mindspeed.core.optimizer.utils import _distributed_group_rank


def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
    """Load chained distributed optimizer parameter states through CPU."""
    if len(self.chained_optimizers) == 1:
        self.chained_optimizers[0].load_parameter_state(filename, update_legacy_format=update_legacy_format)
        return

    states = None
    for idx, optimizer in enumerate(self.chained_optimizers):
        if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
            continue

        if _distributed_group_rank(optimizer.data_parallel_group) == 0 and states is None:
            states = torch.load(filename, map_location='cpu')

        state_dict = states[idx] if states else None
        optimizer.load_parameter_state_from_dp_zero(state_dict, update_legacy_format=update_legacy_format)
