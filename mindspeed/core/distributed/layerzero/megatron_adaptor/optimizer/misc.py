from typing import List, Optional
from functools import wraps

import torch
from megatron.core import parallel_state
from megatron.training.global_vars import get_args
from mindspeed.core.distributed.layerzero.zero3 import LayerZeRO3


def scale_gradients(model, scaling_factor: float):
    if not (isinstance(model, LayerZeRO3) and model._is_root):
        raise ValueError(f"This func expects to be called on a LayerZeRO3 root instance, got {type(model)}")
    
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param.grad.data *= scaling_factor

    
def mga_finalize_model_grads_wrapper(finalize_model_grads):
    @wraps(finalize_model_grads)
    def wrapper(*args, **kwargs):
        global_args = get_args()
        if getattr(global_args, 'layerzero', False):
            num_tokens = kwargs.get('num_tokens', None)
            if num_tokens is not None:
                # the number of tokens is only present on the last stage, so broadcast it
                # to the other ranks in the pipeline parallel group.
                torch.distributed.broadcast(
                    num_tokens,
                    src=parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                # all-reduce across DP ranks.
                torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
                model = kwargs.get('model', None)
                if model is None and args:
                    model = args[0]
                for model_chunk in model:
                    if num_tokens > 0:
                        scaling = 1.0 / num_tokens
                        scale_gradients(model_chunk, scaling)
            return None
        else:
            return finalize_model_grads(*args, **kwargs)

    return wrapper
                