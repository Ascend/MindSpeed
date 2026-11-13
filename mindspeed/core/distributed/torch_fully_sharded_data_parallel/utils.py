# Copyright 2023 Huawei Technologies Co., Ltd.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
from typing import Optional, Union

import torch
from torch.distributed import ProcessGroup

try:
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False


def get_data_parallel_group_if_dtensor(
    tensor: Union[torch.Tensor, "DTensor"], data_parallel_group: "ProcessGroup" = None
) -> Optional["ProcessGroup"]:
    """Gets the data parallel group of the given tensor if it is a DTensor."""
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        device_mesh = tensor.device_mesh
        if device_mesh.ndim == 1:
            current_group = tensor.device_mesh.get_group()
        elif device_mesh.ndim == 2:
            current_group = tensor.device_mesh.get_group(mesh_dim=1)
        else:
            raise ValueError(f"Not support ndims for {device_mesh.ndim}")
        return current_group
    return None
