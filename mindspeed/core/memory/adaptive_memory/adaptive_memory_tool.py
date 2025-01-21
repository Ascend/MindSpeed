# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from enum import Enum, IntEnum
from typing import Set, Dict
import pickle
import torch
from megatron.core import parallel_state as ps
from megatron.training import print_rank_0

BYTES_PER_MB = 1024 * 1024


class LayerAction(IntEnum):
    FULL_RECOMPUTE = 0
    FULL_SWAP = 1
    ADAPTIVE = 2
    NONE = 3


class ModuleAction(IntEnum):
    RECOMPUTE = 0
    SWAP = 1
    NONE = 2


class SingletonBase(type):
    singleton_instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.singleton_instances:
            instance = super().__call__(*args, **kwargs)
            cls.singleton_instances[cls] = instance
        return cls.singleton_instances[cls]


class ContextKey(SingletonBase):
    NAME = "name" # module name
    DEEP = "deep" # module depth
    PREFIX_NAME = "prefix_name" # module parent name
    MODULE = "module" # module field
    SUBMODULES = "submodules" # children modules
    INPUT = "input" # input args total size
    MEMORY = "memory" # current module's activation memory + input + output
    OUTPUT = "output" # output total size
    FORWARD_CNT = "forward_cnt" # forward called times
    PRE_TOTAL_TIME = "pre_total_time" # forward called total time
    AVG_TIME = "avg_time" # forward called avg time
    ALLOWED_ADAPT = "allowed_adapt" # allowed adapted modules, user init and set name at startup
    IS_FUNCTION = "is_function" # mark module if it is a torch.autograd.Function
    IS_MODULE_LIST = "is_module_list" # mark module if it is a torch.nn.ModuleList
    IS_ADAPT_LAYER = "is_adapt_layer" # mark self or parent if self is ALLOWED_ADAPT
    USED_MEM = "used_mem" # model memory consumption, as allocated memory
    DEVICE_MEMORY = "device_memory" # device total memory
    # adaptive
    MODULE_FORWARD_TOTAL_TIME = "module_forward_total_time" # module forward called total time
    MODULE_FORWARD_AVG_TIME = "module_forward_avg_time" # module forward called avg time
    MODULE_FORWARD_CNT = "module_forward_cnt" # module forward called times
    MODULE_SWAP_TOTAL_TIME = "module_swap_total_time" # module forward called total swap time
    MODULE_SWAP_AVG_TIME = "module_swap_avg_time" # module forward called avg swap time
    MODULE_SWAP_CNT = "module_swap_cnt" # module forward called swap times
    MODULE_SWAP_TOTAL_MEMORY = "module_swap_total_memory" # module forward swap total memory
    MODULE_SWAP_AVG_MEMORY = "module_swap_avg_memory" # module forward swap avg memory
    IS_SWAP = "is_swap" # mark module if it is swap
    IS_LAYER0_OF_MODULE0 = "is_layer0_of_module0" # mark module if it is layer0 of module0
    IS_MODLUE_OF_LAYER0 = "is_modlue_of_layer0" # mark module if it belongs to layer0 of module0


class FuncLocationMgr(metaclass=SingletonBase):
    def __init__(self):
        self._module_names = []
        self._function_in_stack = None
        self._function_child = None
        self.is_first_layer = False

    def push_name(self, prefix, name):
        self._module_names.append(f"{prefix}.{name}")
        if self._function_in_stack and not self._function_child:
            self._function_child = f"{prefix}.{name}"

    def pop_name(self, prefix, name):
        last_name = self._module_names.pop()
        if f"{prefix}.{name}" != last_name:
            raise ValueError(f"unexpected module name in stack, expect:{prefix}.{name}, find:{last_name}")

    def get_latest_name(self):
        return self._module_names[-1]

    def set_function_in_stack(self):
        self._function_in_stack = True

    def get_function_location(self, parent):
        if not self._function_child:
            direct_child = ""
        else:
            first_child = self._function_child[len(parent):]
            direct_child = first_child.split(".")[1]
        self._function_child = None
        self._function_in_stack = False
        return direct_child


class AdaptiveStepMgr(metaclass=SingletonBase):
    def __init__(self):
        self.cur_step = 1
        self.skip_steps = 3
        self.recompute_profiling_steps = 0
        self.layer_profiling_steps = 5
        self.swap_profiling_steps = 0
        self.pre_steps = 0

    def init_steps(self, recompute_profiling_steps, swap_profiling_steps):
        self.recompute_profiling_steps = recompute_profiling_steps
        self.swap_profiling_steps = swap_profiling_steps
        self.pre_steps = self.skip_steps + recompute_profiling_steps + swap_profiling_steps + self.layer_profiling_steps

    def get_cur_step(self):
        return self.cur_step

    def reset_step(self, step_num):
        self.cur_step = step_num

    def incr_step(self):
        self.cur_step += 1

    def is_skipping_step(self):  # 两处调用，profiling时决定是否下发event，step里是否return
        return self.cur_step <= self.skip_steps

    def is_recompute_profiling_step(self):
        pre_steps = self.skip_steps
        return pre_steps < self.cur_step <= pre_steps + self.recompute_profiling_steps

    def is_last_recompute_profiling_step(self):
        return self.cur_step == (self.skip_steps + self.recompute_profiling_steps)

    def is_layer_profiling_step(self):
        pre_steps = self.skip_steps + self.recompute_profiling_steps
        return pre_steps < self.cur_step <= pre_steps + self.layer_profiling_steps

    def is_last_layer_profiling_step(self):
        return self.cur_step == self.skip_steps + self.recompute_profiling_steps + self.layer_profiling_steps

    def is_layer_profiling_done(self):
        return self.cur_step >= self.skip_steps + self.recompute_profiling_steps + self.layer_profiling_steps

    def is_all_profiling_done(self):    # note: this called in step_func, should use > instead of >=
        return self.cur_step > self.pre_steps

    def is_swap_profiling_step(self):
        pre_steps = self.skip_steps + self.recompute_profiling_steps + self.layer_profiling_steps
        return pre_steps < self.cur_step <= self.pre_steps

    def is_swap_profiling_done(self):   # note: this called after step_func, should use >= instead of >
        return self.cur_step >= self.pre_steps


class ForwardCounter(metaclass=SingletonBase):
    def __init__(self):
        self._counter: int = 0

    def get_count(self):
        return self._counter

    def incr_cnt(self):
        self._counter += 1


class FuncLocation:
    def __init__(self, idx: int, func_name: str, action: ModuleAction):
        self.layer_idx = idx
        self.func_name = func_name
        self.action = action


class CpuTensorCache(metaclass=SingletonBase):
    def __init__(self):
        self.shape_to_tensor_list_map: Dict[(torch.Size, torch.dtype), Set[torch.Tensor]] = {}

    def get_cpu_tensor(self, shape: torch.Size, dtype: torch.dtype):
        tensor_set = self.shape_to_tensor_list_map.setdefault((shape, dtype), set())
        if len(tensor_set) != 0:
            cpu_tensor = tensor_set.pop()
        else:
            cpu_tensor = torch.empty(shape, dtype=dtype, pin_memory=True, device='cpu')
        return cpu_tensor

    def release_cpu_tensor(self, cpu_tensor):
        tensor_set = self.shape_to_tensor_list_map.setdefault((cpu_tensor.shape, cpu_tensor.dtype), set())
        tensor_set.add(cpu_tensor)


def broadcast_in_mp_dp(tensor, src, mp, dp):
    if mp > 1 and ps.get_tensor_model_parallel_src_rank() == src:
        broadcast_src = ps.get_tensor_model_parallel_src_rank()
        broadcast_group = ps.get_tensor_model_parallel_group()
        torch.distributed.broadcast(tensor, src=broadcast_src, group=broadcast_group)
    if dp > 1:
        broadcast_src = ps.get_data_parallel_src_rank(True)
        broadcast_group = ps.get_data_parallel_group(True)
        torch.distributed.broadcast(tensor, src=broadcast_src, group=broadcast_group)


def broadcast_obj(obj):
    mp = ps.get_tensor_model_parallel_world_size()
    dp = ps.get_data_parallel_world_size(True)

    global_rank = torch.distributed.get_rank()
    src = (global_rank // (mp * dp)) * dp * mp
    obj_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
    obj_shape_tensor = torch.tensor(obj_tensor.shape, device=torch.npu.current_device())
    broadcast_in_mp_dp(obj_shape_tensor, src, mp, dp)
    obj_len = obj_shape_tensor.cpu().tolist()
    if global_rank == src:
        obj_tensor_npu = obj_tensor.npu()
    else:
        obj_tensor_npu = torch.empty(obj_len, dtype=torch.uint8, device=torch.npu.current_device())
    broadcast_in_mp_dp(obj_tensor_npu, src, mp, dp)
    result = pickle.loads(obj_tensor_npu.cpu().numpy().tobytes())
    del obj_tensor_npu
    return result