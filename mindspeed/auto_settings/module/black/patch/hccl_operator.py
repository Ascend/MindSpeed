from functools import wraps
from typing import Any, Literal

import torch
from megatron.training.global_vars import get_args

from mindspeed.auto_settings.module.black.auto_patch import AutoPatcher

Block_Name = Literal['attention', 'attention_grad', 'moe/mlp', 'moe/mlp_grad', 'else']


_BLOCK_NAME = 'else'
_COLLECTED_BLOCK_NAMES = {}


def set_block_name(block_name: Block_Name):
    global _BLOCK_NAME, _COLLECTED_BLOCK_NAMES
    _BLOCK_NAME = block_name

    if block_name not in _COLLECTED_BLOCK_NAMES:
        _COLLECTED_BLOCK_NAMES[block_name] = 1
    else:
        _COLLECTED_BLOCK_NAMES[block_name] += 1


def get_communication_info(comm_type, *args, **kwargs):
    output_shape = list(args[0].shape)

    if 'allReduce' in comm_type:
        input_shape = list(args[0].shape)
    else:
        input_shape = list(args[1].shape)

    data_type = str(args[0].dtype)
    async_op = kwargs.get('async_op', False)
    group = kwargs.get('group', None)
    world_size = torch.distributed.get_world_size(group=group) if group else ''

    return {
        'input_shapes': input_shape,
        'output_shapes': output_shape,
        'world_size': world_size,
        'async': async_op,
        'dtype': data_type
    }


def hccl_operator_decorator(fn_name, fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        global _BLOCK_NAME, _COLLECTED_BLOCK_NAMES

        prof_file = get_args().prof_file
        if prof_file and _BLOCK_NAME != 'else' and _COLLECTED_BLOCK_NAMES.get(_BLOCK_NAME, 0) <= 1:
            auto_profiler = AutoPatcher(prof_file)

            if 'comm_operators' not in auto_profiler.context:
                auto_profiler.context['comm_operators'] = {}

            if _BLOCK_NAME not in auto_profiler.context['comm_operators']:
                auto_profiler.context['comm_operators'][_BLOCK_NAME] = []
            
            operator = get_communication_info(fn_name, *args, **kwargs)
            operator['type'] = fn_name

            auto_profiler.context['comm_operators'][_BLOCK_NAME].append(operator)

        return fn(*args, **kwargs)
    
    return wrapper


def p2p_operator_decorator(fn_name, fn):
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global _BLOCK_NAME, _COLLECTED_BLOCK_NAMES

        prof_file = get_args().prof_file
        if prof_file and _BLOCK_NAME != 'else' and _COLLECTED_BLOCK_NAMES.get(_BLOCK_NAME, 0) <= 1:
            auto_profiler = AutoPatcher(prof_file)

            if 'comm_operators' not in auto_profiler.context:
                auto_profiler.context['comm_operators'] = {}

            if _BLOCK_NAME not in auto_profiler.context['comm_operators']:
                auto_profiler.context['comm_operators'][_BLOCK_NAME] = []

            operator = {}
            operator['type'] = fn_name
            operator['input_shapes'] = args[0].shape
            operator['output_shape'] = args[0].shape
            auto_profiler.context['comm_operators'][_BLOCK_NAME].append(operator)

        return fn(*args, **kwargs)
    
    return wrapper


class AttentionStartOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        set_block_name('attention')
        output = input_data * 1
        return output
              
    @staticmethod
    def backward(ctx, grad_output):
        set_block_name('else')
        return grad_output
    

class AttentionEndOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        set_block_name('else')
        output = input_data * 1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        set_block_name('attention_grad')
        return grad_output


class MOEOrMLPStartOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        set_block_name('moe/mlp')
        output = input_data * 1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        set_block_name('else')
        return grad_output
    

class MOEOrMLPEndOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        set_block_name('else')
        output = input_data * 1
        return output
    
    @staticmethod
    def backward(ctx, grad_output) -> Any:
        set_block_name('moe/mlp_grad')
        return grad_output