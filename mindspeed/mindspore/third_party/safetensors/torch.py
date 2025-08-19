# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import os 
import sys
from typing import Dict, Optional, Union

import torch

_float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
_float8_e5m2 = getattr(torch, "float8_e5m2", None)

_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
    _float8_e4m3fn: 1,
    _float8_e5m2: 1,
}


def storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0
        
        
def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.nbytes
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _SIZE[tensor.dtype]
        

def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    torch.serialization.safe_save_file(tensors, filename, metadata=metadata)
    
    
def load_file(filename: Union[str, os.PathLike], device: Union[str, int] = "cpu") -> Dict[str, torch.Tensor]:
    return torch.serialization.safe_load_file(filename, device=device)
