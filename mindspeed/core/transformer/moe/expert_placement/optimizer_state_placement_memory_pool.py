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


class OptimizerStatePlacementMemoryPool:
    def __init__(self):
        self.pool = {}  
        self.total_allocated = 0  

    def get(self, shape, dtype, init_method='zero'):
        key = (tuple(shape), dtype)
        
        if key in self.pool and self.pool[key]:
            tensor = self.pool[key].pop()
            if init_method == 'zero':
                tensor.zero_()
            elif init_method == 'normal':
                tensor.normal_()
        else:
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            self.total_allocated += 1
            if init_method == 'zero':
                tensor.zero_()
            elif init_method == 'normal':
                tensor.normal_()
            self.put(tensor)
        return tensor

    def put(self, tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self.pool:
            self.pool[key] = []
        self.pool[key].append(tensor.detach())

    def clear(self, empty_cache=True):
        self.pool.clear()
        self.total_allocated = 0
        if not empty_cache:
            return
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.synchronize()
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            import warnings
            warnings.warn(f"Failed to clear device cache: {str(e)}", RuntimeWarning)
