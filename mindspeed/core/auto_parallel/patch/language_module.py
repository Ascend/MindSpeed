# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import time
from functools import wraps

import torch
from megatron.training import get_args


def compute_language_model_loss_wrapper(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        prof_file = get_args().prof_file
        if prof_file:
            from ..utils.profiler import AutoProfiler
            auto_profiler = AutoProfiler(prof_file)

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
