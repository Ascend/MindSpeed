# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import os 
import sys
from typing import Dict, Optional, Union

import torch
 

def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    torch.serialization.safe_save_file(tensors, filename, metadata=metadata)
    
    
def load_file(filename: Union[str, os.PathLike], device: Union[str, int] = "cpu") -> Dict[str, torch.Tensor]:
    return torch.serialization.safe_load_file(filename, device=device)
