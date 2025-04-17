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
