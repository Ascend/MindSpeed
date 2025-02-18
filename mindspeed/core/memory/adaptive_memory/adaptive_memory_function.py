# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from copy import copy
from typing import List, Any

import torch
from megatron.core.tensor_parallel.random import checkpoint

from megatron.training import print_rank_0, get_args
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core import parallel_state as ps

from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
from .adaptive_memory_profiling import AdaptiveMemoryProfiling
from .adaptive_memory_solver import AdaptMemGraphSolver
from .adaptive_memory_prefetch import AdaptiveMemoryPrefetch, pre_forward_func
from .adaptive_memory_tool import AdaptiveStepMgr, SingletonBase, ModuleAction, BYTES_PER_MB, ContextKey as Key
from .adaptive_memory_tool import FuncLocationMgr, ForwardCounter
from .adaptive_memory_swap_manager import SwapManager


class FunctionCtxMgr(metaclass=SingletonBase):
    def __init__(self):
        self._ctx_dict = {}
        self._child_dict = {}

    def update_ctx(self, func_name, new_ctx, child_name):
        if func_name not in self._ctx_dict:
            self._ctx_dict[func_name] = new_ctx
            self._ctx_dict[func_name][Key.FORWARD_CNT] = 1
            self._ctx_dict[func_name][Key.AVG_TIME] = new_ctx[Key.PRE_TOTAL_TIME]
            self._ctx_dict[func_name][Key.IS_FUNCTION] = True
        else:
            target_ctx = self._ctx_dict[func_name]
            target_ctx[Key.FORWARD_CNT] += 1
            target_ctx[Key.PRE_TOTAL_TIME] += new_ctx[Key.PRE_TOTAL_TIME]
            target_ctx[Key.AVG_TIME] = target_ctx[Key.PRE_TOTAL_TIME] / target_ctx[Key.FORWARD_CNT]

        if func_name not in self._child_dict:
            self._child_dict[func_name] = child_name

    def ctx_iter(self):
        for key in self._ctx_dict.keys():
            yield self._ctx_dict.get(key), self._child_dict.get(key)


class FunctionProfilingWrapper:
    def __init__(self, function):
        self._function = function
        self._ctx = {Key.NAME: function.__name__}

        self.start_event = torch.npu.Event(enable_timing=True)
        self.end_evnet = torch.npu.Event(enable_timing=True)

    def _pre_process(self, *args):
        self._ctx[Key.PREFIX_NAME] = FuncLocationMgr().get_latest_name()
        self._ctx[Key.DEEP] = len(self._ctx[Key.PREFIX_NAME].split("."))
        self._ctx[Key.IS_MODLUE_OF_LAYER0] = True
        FuncLocationMgr().set_function_in_stack()

        self._ctx[Key.INPUT] = AdaptiveMemoryProfiling().cal_input_output_size(args) / BYTES_PER_MB
        self._ctx[Key.MEMORY] = torch.npu.memory_allocated() - self._ctx[Key.INPUT]
        self.start_event.record()

    def _post_process(self, outputs):
        self.end_evnet.record()
        torch.npu.synchronize()
        self._ctx[Key.PRE_TOTAL_TIME] = self.start_event.elapsed_time(self.end_evnet)
        self._ctx[Key.OUTPUT] = AdaptiveMemoryProfiling().cal_input_output_size(outputs) / BYTES_PER_MB
        self._ctx[Key.MEMORY] = (torch.npu.memory_allocated() - self._ctx[Key.MEMORY]) / BYTES_PER_MB

        child_name = FuncLocationMgr().get_function_location(self._ctx[Key.PREFIX_NAME])
        FunctionCtxMgr().update_ctx(self._function.__name__, self._ctx, child_name)

    def run_profiling(self, *args, **kwargs):
        self._pre_process(args)
        outputs = self._function.apply(*args, **kwargs)
        self._post_process(outputs)
        return outputs


def pack_hook(tensor):
    return SwapManager().prefetch_pack(tensor)


def unpack_hook(swap_tensor):
    return SwapManager().prefetch_unpack(swap_tensor)


def pre_profiling_process(module_name):
    pre_forward_func(module_name, False)


def post_profiling_process(module_name):
    AdaptiveMemoryPrefetch().sync_d2h_for_recording_time(module_name, True)


def wrap_swap_profiling(function, module_name, *args):
    pre_profiling_process(module_name)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        outputs = function.apply(*args)
    post_profiling_process(module_name)
    return outputs


def wrap_function(function, *args):
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        return function.apply(*args)


def adapt_mem_func_wrapper(fc_class, *args):
    if not issubclass(fc_class, torch.autograd.Function):
        raise TypeError("adapt_mem_func_wrapper only support subclass of torch.autograd.Function")
    cnt = ForwardCounter().get_count()
    is_first_layer = FuncLocationMgr().is_first_layer
    if AdaptiveStepMgr().is_recompute_profiling_step() and is_first_layer:
        if fc_class.__name__ not in AdaptiveMemoryPrefetch().function_list:
            AdaptiveMemoryPrefetch().function_list.append(fc_class.__name__)
        return FunctionProfilingWrapper(fc_class).run_profiling(*args)
    elif AdaptiveStepMgr().is_swap_profiling_step() and is_first_layer: # recording swap profiling
        if FunctionCtxMgr()._ctx_dict.get(fc_class.__name__)[Key.DEEP] == AdaptiveMemoryPrefetch().function_swap_profiling_deep:
            module_full_name = FunctionCtxMgr()._ctx_dict.get(fc_class.__name__)[Key.PREFIX_NAME] + "." + fc_class.__name__
            return wrap_swap_profiling(fc_class, module_full_name, *args)
    elif AdaptiveStepMgr().is_swap_profiling_done() and not AdaptiveMemoryPrefetch().is_stable_apply and is_first_layer:
        if fc_class.__name__ in AdaptiveMemoryPrefetch().prefetch_function_list:
            return wrap_function(fc_class, *args)
    elif AdaptiveStepMgr().is_all_profiling_done() and AdaptiveMemoryPrefetch().is_stable_apply: # do one of prefetch/recompute/swap
        action = AdaptMemGraphSolver().get_func_action(fc_class.__name__, cnt - 1)
        if action == ModuleAction.RECOMPUTE:
            def fc_class_apply():
                return fc_class.apply(*args)
            return checkpoint(fc_class_apply, False)
        elif action == ModuleAction.SWAP:
            return wrap_function(fc_class, *args)
    return fc_class.apply(*args) # do default function.apply