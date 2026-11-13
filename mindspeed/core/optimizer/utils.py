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

import torch


def _to_step_int(step):
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)


def _distributed_group_rank(group):
    if hasattr(group, 'rank') and callable(group.rank):
        return group.rank()
    return torch.distributed.get_rank(group)
