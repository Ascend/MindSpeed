# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
