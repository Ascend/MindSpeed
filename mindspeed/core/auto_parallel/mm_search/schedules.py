# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import time
from functools import wraps

import torch

from megatron.training import get_args
from mindspeed.core.auto_parallel.mm_search.help import PROFILE_CONTENT


def backward_step_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global_args = get_args()
        if global_args.auto_parallel_profile:
            # get model backward time
            torch.npu.synchronize()
            st_time = time.time()
            grad = fn(*args, **kwargs)
            torch.npu.synchronize()
            PROFILE_CONTENT["bwd_time"].append((time.time() - st_time) * 1000)
        else:
            grad = fn(*args, **kwargs)
        return grad
    return wrapper

