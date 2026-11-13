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

from functools import wraps

import torch
from megatron.training.global_vars import get_args

from .hccl_operator import MOEOrMLPStartOp, MOEOrMLPEndOp


def moelayer_forward_decorator(fn):

    @wraps
    def wrapper(*args, **kwargs):
        prof_file = get_args().prof_file
        if prof_file:
            args[1] = MOEOrMLPStartOp.apply(args[1])
            activation_func_1 = torch.nn.Softplus()
            args[1] = activation_func_1(args[1])

            output, mlp_bias = fn(*args, **kwargs)

            activation_func_2 = torch.nn.Softshrink()
            output = activation_func_2(output)
            output = MOEOrMLPEndOp.apply(output)
            return output, mlp_bias
        
        return fn(*args, **kwargs)
    
    return wrapper
