# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from functools import lru_cache
from typing import Any, Tuple, Union
import torch


def get_torch_storage_size(tensor: "torch.Tensor") -> int:
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            # this returns the size of the FULL tensor in bytes
            return tensor.nbytes
    except ImportError:
        pass

    try:
        # for torch 2.1 and above we can also handle tensor subclasses
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            attrs, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            return sum(get_torch_storage_size(getattr(tensor, attr)) for attr in attrs)
    except ImportError:
        # for torch version less than 2.1, we can fallback to original implementation
        pass

    try:
        return tensor.nbytes
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _get_dtype_size(tensor.dtype)
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _get_dtype_size(tensor.dtype)
        

def storage_ptr(tensor: "torch.Tensor") -> Union[int, Tuple[Any, ...]]:
    try:
        # for torch 2.1 and above we can also handle tensor subclasses
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            return _get_unique_id(tensor)  # type: ignore
    except ImportError:
        # for torch version less than 2.1, we can fallback to original implementation
        pass

    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor
        except NotImplementedError:
            # Fallback for meta storage
            return 0
        

@lru_cache()
def _get_dtype_size(dtype: "torch.dtype") -> int:
    # torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
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
    return _SIZE[dtype]