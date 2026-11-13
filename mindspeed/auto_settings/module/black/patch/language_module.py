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

import time
from functools import wraps

import torch
from megatron.training import get_args

from mindspeed.auto_settings.module.black.auto_patch import AutoPatcher


def compute_language_model_loss_wrapper(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        prof_file = get_args().prof_file
        if prof_file:
            auto_profiler = AutoPatcher(prof_file)

            if 'loss' not in auto_profiler.context:
                auto_profiler.context['loss'] = {}
                
            torch.cuda.synchronize()
            used_mem, _ = auto_profiler.get_memory_status()
            start_time = time.time()
            loss = fn(*args, **kwargs)
            torch.cuda.synchronize()
            loss_time = (time.time() - start_time) * 1000
            cur_used_mem, cur_max_mem = auto_profiler.get_memory_status()
            auto_profiler.context['loss']['time'] = loss_time
            auto_profiler.context['loss']['memory'] = (cur_used_mem - used_mem) / auto_profiler.unit_gb
            auto_profiler.context['loss']['max_memory'] = (cur_max_mem - used_mem) / auto_profiler.unit_gb
            return loss
        
        return fn(*args, **kwargs)
    
    return wrapper
