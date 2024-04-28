import sys
import time
from copy import deepcopy
from functools import wraps
from collections.abc import Iterable

import acl
import numpy as np
import torch
import torch.nn
import torch_npu

from megatron import print_rank_0, get_num_microbatches
from megatron import get_args
from megatron.core import parallel_state

from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute_apply import get_recompute_hook
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute_apply import get_swap_hook
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute_apply import register_recursive_apply as apply_adaptive_recompute
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute_solver import get_graph_solver, GraphSolver
from ascendspeed.core.memory.adaptive_recomputing.swap_manager import SwapManager, get_tensor_mem_size


class AdaptiveRecomputePolicy:
    adaptive_recomputing_policy = None

    def __init__(self):
        # total swap out size after OOM
        self.record_swap_out_size = 0
        # module context copy
        self.context_copy = None
        # unit for device memory size(MB)
        self.unit_mb = 1024 * 1024
        # find target device memory for policy
        self.is_find_target_device_memory = False
        # swap size for this step OOM
        self.swap_size = 0

        # policy
        self.cur_recompute_policy = []
        self.oom_recompute_policy_list = []
        self.normal_recompute_policy_list = []

        # device memory dichotomy for solve graph
        self.device_memory_dichotomy_left = 0
        self.device_memory_dichotomy_right = 0
        self.cur_device_memory = -1
        self.stop_dichotomy_value = 1

        # device memory free default is maxsize
        self.default_device_memory = sys.maxsize
        all_args = get_args()
        if all_args.adaptive_recompute_device_size >= 0:
            self.default_device_memory = all_args.adaptive_recompute_device_size
        self.hccl_memory = 0

        self.remove_swap_manager_hook_step = 0
        self.last_num_alloc_retries = torch.npu.memory_stats()["num_alloc_retries"]
        self.change_num_alloc_retries_times = 0
        self.first_non_oom_device_memory = 0
        self.check_non_oom_times = 0

    @staticmethod
    def tensor_all_reduce(num_list, op):
        shard_tensor = torch.tensor(num_list, device=torch.npu.current_device())
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                shard_tensor,
                op=op,
                group=parallel_state.get_tensor_model_parallel_group(), )
        if parallel_state.get_data_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                shard_tensor,
                op=op,
                group=parallel_state.get_data_parallel_group(), )
        result = shard_tensor.cpu().numpy().tolist()
        del shard_tensor
        return result

    @staticmethod
    def is_policy_in_list(policy, policy_list):
        for p in policy_list:
            if np.all(p == policy):
                return True
        return False

    def is_stable_policy(self, profiling_step):
        all_args = get_args()
        # not activate swap function or remove swap manager hook
        if not all_args.adaptive_recompute_device_swap or (profiling_step > self.remove_swap_manager_hook_step != 0):
            return True

        total_swap_out_size = SwapManager().total_swap_out_size
        self.swap_size = (total_swap_out_size - self.record_swap_out_size) // self.unit_mb
        self.check_num_alloc_retries()
        num_list = [
            int(total_swap_out_size), int(self.hccl_memory), int(self.swap_size),
            int(self.is_find_target_device_memory), int(self.change_num_alloc_retries_times)
        ]
        size_tensor = self.tensor_all_reduce(num_list, torch.distributed.ReduceOp.MAX)
        total_swap_out_size = size_tensor[0]
        self.hccl_memory = size_tensor[1]
        self.swap_size = size_tensor[2]
        self.is_find_target_device_memory = bool(size_tensor[3])
        self.change_num_alloc_retries_times = size_tensor[4]
        SwapManager().total_swap_out_size = total_swap_out_size

        if self.swap_size <= 0 and self.is_find_target_device_memory:
            return True
        self.record_swap_out_size = total_swap_out_size
        return False

    def get_default_device_memory(self, max_device_memory):
        self.default_device_memory = min(self.default_device_memory, max_device_memory)
        size_tensor = self.tensor_all_reduce([int(self.default_device_memory)], torch.distributed.ReduceOp.MIN)
        self.default_device_memory = size_tensor[0]

    def check_cur_recompute_policy(self):
        if len(self.cur_recompute_policy) == 0:
            return
        is_exist_oom = self.is_policy_in_list(self.cur_recompute_policy, self.oom_recompute_policy_list)
        is_exist_normal = self.is_policy_in_list(self.cur_recompute_policy, self.normal_recompute_policy_list)
        if self.swap_size > 0:
            if not is_exist_oom:
                self.oom_recompute_policy_list.append(deepcopy(self.cur_recompute_policy))
            if is_exist_normal:
                self.normal_recompute_policy_list.remove(self.cur_recompute_policy)
            return
        if is_exist_oom or self.change_num_alloc_retries_times != 0:
            return
        if not is_exist_normal:
            self.normal_recompute_policy_list.append(deepcopy(self.cur_recompute_policy))

    def dichotomy_best(self):
        # last policy is instability
        if self.is_find_target_device_memory:
            self.device_memory_dichotomy_left = self.first_non_oom_device_memory
            self.device_memory_dichotomy_right = self.cur_device_memory
        self.is_find_target_device_memory = False
        if self.cur_device_memory == -1:
            return self.default_device_memory

        # OOM
        if self.swap_size > 0:
            self.check_non_oom_times = 0
            self.change_num_alloc_retries_times = 0
            self.device_memory_dichotomy_right = self.cur_device_memory
            if self.first_non_oom_device_memory >= self.cur_device_memory:
                self.first_non_oom_device_memory = 0
            if self.device_memory_dichotomy_right <= self.device_memory_dichotomy_left:
                self.device_memory_dichotomy_left = 0
            return (self.device_memory_dichotomy_left + self.device_memory_dichotomy_right) // 2

        # check non oom policy
        if self.change_num_alloc_retries_times != 0 and self.check_non_oom_times == 0:
            print_rank_0(f"current policy may be an unstable one, try to check it once again, "
                         f"policy device memory: {self.cur_device_memory}")
            self.check_non_oom_times += 1
            self.change_num_alloc_retries_times = 0
            return self.cur_device_memory

        self.check_non_oom_times = 0
        self.change_num_alloc_retries_times = 0
        self.device_memory_dichotomy_left = self.cur_device_memory
        if self.first_non_oom_device_memory == 0:
            self.first_non_oom_device_memory = self.cur_device_memory
        if self.device_memory_dichotomy_right - self.device_memory_dichotomy_left <= self.stop_dichotomy_value:
            self.is_find_target_device_memory = True
            return self.device_memory_dichotomy_left

        return (self.device_memory_dichotomy_left + self.device_memory_dichotomy_right) // 2

    def solve_recompute_policy(self, profiling_step):
        is_known_policy = True
        self.remove_swap_manager_hook_step = profiling_step + 1
        swap_size = self.swap_size
        recompute_policy_list = None
        while is_known_policy:
            torch.npu.synchronize()
            self.cur_device_memory = self.dichotomy_best()
            if self.check_non_oom_times == 0:
                recompute_policy_list = get_graph_solver().get_policy(self.cur_device_memory)
                np_result = np.array(recompute_policy_list)
                self.cur_recompute_policy = np.array([r * r[0] for r in np_result]).sum(axis=0).tolist()
            if self.is_find_target_device_memory:
                self.remove_swap_manager_hook_step = profiling_step + 10
                print_rank_0(
                    f"success to find the target value of the current round of search: {self.cur_device_memory}")
                break
            # OOM policy
            if self.is_policy_in_list(self.cur_recompute_policy, self.oom_recompute_policy_list):
                self.swap_size = max(self.swap_size, 1)
                continue
            # no OOM policy
            if self.is_policy_in_list(self.cur_recompute_policy, self.normal_recompute_policy_list):
                self.swap_size = 0
                continue
            is_known_policy = False
        if recompute_policy_list is None:
            print_rank_0(f"{get_graph_solver().final_policy_info}")
            return None
        get_graph_solver().print_list_to_policy(recompute_policy_list)
        print_rank_0(
            f"max available memory: {self.context_copy['max_device_memory']}, previous policy swap size: {swap_size}, "
            f"next policy device memory: {self.cur_device_memory}")
        print_rank_0(f"{get_graph_solver().without_recompute_info}\n{get_graph_solver().all_recompute_info}\n"
                     f"{get_graph_solver().selective_recompute_info}\n{get_graph_solver().final_policy_info}")
        return self.set_tag_to_context(recompute_policy_list)

    def set_tag_to_context(self, recompute_policy_list):
        context = deepcopy(self.context_copy)
        SwapManager().reset_policy_peak_memory()
        solver = GraphSolver()
        solver.layer_full_recompute_combination = get_graph_solver().layer_full_recompute_combination
        solver.layer_without_recompute_combination = get_graph_solver().layer_without_recompute_combination
        solver.layer_recompute_one_combination = get_graph_solver().layer_recompute_one_combination
        solver.layers_combination = get_graph_solver().layers_combination
        solver.get_layers_module(context, "")
        solver.get_no_recompute_layer()
        solver.apply_policy_to_model(recompute_policy_list)
        return context

    def check_num_alloc_retries(self):
        num_alloc_retries = torch.npu.memory_stats()["num_alloc_retries"]
        if num_alloc_retries == self.last_num_alloc_retries:
            return
        retries_times = num_alloc_retries - self.last_num_alloc_retries
        self.last_num_alloc_retries = num_alloc_retries
        if self.swap_size == 0 and (retries_times > 1 or self.check_non_oom_times != 0):
            self.swap_size = 1
        if self.swap_size > 0:
            return

        self.change_num_alloc_retries_times += 1
        if self.change_num_alloc_retries_times > 1:
            print_rank_0(f"[^?^?^] this is a unstable policy, try select another one.")
            self.swap_size = 1


def get_adaptive_recomputing_policy():
    if AdaptiveRecomputePolicy.adaptive_recomputing_policy is None:
        AdaptiveRecomputePolicy.adaptive_recomputing_policy = AdaptiveRecomputePolicy()
    return AdaptiveRecomputePolicy.adaptive_recomputing_policy


class MemoryPeakPrediction:
    memory_peak_prediction = None

    def __init__(self):
        self.previous_layer_info = {}
        self.record_max_memory_allocated = {}
        self.memory_fragment = 0

    def get_peak_memory(self, layer_name, memory_info, is_update_peak):
        _max_memory_allocated = 0
        for k, v in self.record_max_memory_allocated.items():
            _name = k[:str.rfind(k, ".")]
            if _name == layer_name:
                _max_memory_allocated = max(_max_memory_allocated, v)
        if _max_memory_allocated == 0:
            _max_memory_allocated = memory_info["max_memory_allocated"]
        self.record_max_memory_allocated[layer_name] = _max_memory_allocated
        peak_memory = _max_memory_allocated - memory_info['memory']
        if not is_update_peak:
            return peak_memory
        origin_peak = 0
        if layer_name in SwapManager().origin_layers_peak_memory.keys():
            origin_peak = SwapManager().origin_layers_peak_memory[layer_name]
        SwapManager().origin_layers_peak_memory.update({layer_name: max(origin_peak, peak_memory)})
        return peak_memory

    @staticmethod
    def check_forward_mem_info(state, layer_name, memory_info, is_pre_hook=True):
        process_free = memory_info["reserved_memory"] - memory_info["used_memory"]
        state["free"] = memory_info["free"]
        state["process_free"] = process_free
        peak_memory = 0
        memory_fragment = get_memory_peak_prediction().memory_fragment
        if is_pre_hook and layer_name in SwapManager().policy_peak_memory.keys():
            peak_memory = SwapManager().policy_peak_memory[layer_name]
        if not is_pre_hook and layer_name in SwapManager().layers_interval_peak_memory.keys():
            peak_memory = SwapManager().layers_interval_peak_memory[layer_name]
        if peak_memory <= 0:
            return
        supple_size = peak_memory + memory_fragment - process_free - memory_info["free"]
        if SwapManager().swap_status and supple_size > 0:
            SwapManager().swap_out_by_size(peak_memory, is_prediction=True)

    def record_interval_peak_memory(self, current_ctx, memory_info):
        if not self.previous_layer_info or 'is_recomputing_layer' in current_ctx:
            return
        pre_layer_name = self.previous_layer_info['layer_name']
        interval_peak_memory = memory_info['max_memory_allocated'] - self.previous_layer_info['memory_info'][
            'used_memory']
        module_interval_peak_memory = SwapManager().layers_interval_peak_memory
        old_interval_peak_memory = 0
        if pre_layer_name in module_interval_peak_memory.keys():
            old_interval_peak_memory = module_interval_peak_memory[pre_layer_name]
        interval_peak_memory = max(interval_peak_memory, old_interval_peak_memory)
        module_interval_peak_memory.update({pre_layer_name: interval_peak_memory})

    def record_interval_memory_info(self, layer_name, memory_info):
        self.previous_layer_info["layer_name"] = layer_name
        self.previous_layer_info["memory_info"] = memory_info


def get_memory_peak_prediction():
    if MemoryPeakPrediction.memory_peak_prediction is None:
        MemoryPeakPrediction.memory_peak_prediction = MemoryPeakPrediction()
    return MemoryPeakPrediction.memory_peak_prediction


class AdaptiveRecompute:
    adaptive_recomputing = None

    def __init__(self):
        # layer profiling info
        self.context = {
            'module': []
        }
        #record allowed recomputing module
        self.allowed_recomputing_module = []
        # profiling prefix
        self.profiling_prefix = ""
        # save origin modules
        self.checkpointed_modules = []
        # save modules hook, remove it after apply policy
        self.modules_hooks = []
        # current profiling step
        self.profiling_step = 0
        # step for stop profiling, default is 10
        self.stop_profiling_step = 10
        # min step for stop profiling
        self.min_profiling_step = 5
        # step for solve graph by adaptive recompute, after step for stop profiling
        self.solve_graph_at_step = 11
        # unit for device memory size(MB)
        self.unit_mb = 1024 * 1024
        # pp or vpp
        self.num_warmup_micro_batches = 1

    @staticmethod
    def get_memory_status():
        free, all_memory, _ = acl.rt.get_mem_info(1)
        memory_info = {
            "free": free,
            "all_memory": all_memory,
            "used_memory": torch.npu.memory_allocated(),
            "reserved_memory": torch.npu.memory_reserved(),
            "max_memory_allocated": torch.npu.max_memory_allocated()
        }

        return memory_info

    def get_num_warmup_micro_batches(self, num_model_chunks):
        if parallel_state.get_pipeline_model_parallel_world_size() <= 1:
            return
        num_microbatches = get_num_microbatches()
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        total_num_micro_batches = num_microbatches * num_model_chunks
        if num_model_chunks == 1:
            num_warmup_micro_batches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            num_warmup_micro_batches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_micro_batches += (num_model_chunks - 1) * pipeline_parallel_size
        num_warmup_micro_batches += 1
        self.num_warmup_micro_batches = min(num_warmup_micro_batches, total_num_micro_batches)

    def pre_hook_func(self, state, prefix, name, *args, **kargs):
        torch.npu.synchronize()
        layer_name = prefix + "." + name if prefix != "" else name
        SwapManager().cur_pre_hook_layer_name = layer_name
        SwapManager().cur_post_hook_layer_name = ""
        memory_info = self.get_memory_status()
        get_memory_peak_prediction().record_interval_peak_memory(state, memory_info)
        torch.npu.reset_max_memory_allocated()
        state['memory'] = memory_info['used_memory']
        state['time'] = time.time()
        state['input'] = self._cal_input_output_size(args)
        if self.profiling_step >= self.solve_graph_at_step \
                and not get_adaptive_recomputing_policy().is_find_target_device_memory:
            get_memory_peak_prediction().check_forward_mem_info(state, layer_name, memory_info)

    def post_hook_func(self, state, prefix, name, args, output):
        torch.npu.synchronize()
        layer_name = prefix + "." + name if prefix != "" else name
        SwapManager().cur_pre_hook_layer_name = ""
        SwapManager().cur_post_hook_layer_name = layer_name
        memory_info = self.get_memory_status()
        get_memory_peak_prediction().record_interval_peak_memory(state, memory_info)
        get_memory_peak_prediction().record_interval_memory_info(layer_name, memory_info)
        torch.npu.reset_max_memory_allocated()
        memory_info['memory'] = state['memory']
        state['peak_memory'] = get_memory_peak_prediction().get_peak_memory(layer_name, memory_info,
                                                                            self.profiling_step <= self.solve_graph_at_step)
        output_memory = self._cal_input_output_size(output)
        state['memory_bytes'] = memory_info['used_memory'] - state['memory'] - output_memory
        state['input'] += output_memory
        state['memory'] = state['memory_bytes'] // self.unit_mb
        if 'pre_total_time' in state:
            state['forward_cnt'] += 1
            state['time'] = (time.time() - state['time']) * 1000
            state['pre_total_time'] += state['time']
            try:
                state['time'] = state['pre_total_time'] / state['forward_cnt']
            except ZeroDivisionError:
                state['time'] = 0
        else:
            state['forward_cnt'] = 0
            state['time'] = (time.time() - state['time']) * 1000
            state['pre_total_time'] = 0
        if self.profiling_step < self.solve_graph_at_step:
            return
        peak_memory = state['peak_memory']
        if layer_name in SwapManager().policy_peak_memory.keys():
            peak_memory = SwapManager().policy_peak_memory[layer_name]
        # record fragment
        fragment = max(0, state["process_free"] + (state["free"] - memory_info["free"]) - peak_memory)
        if fragment != 0 and memory_info["free"] < state["free"] and state["process_free"] >= peak_memory:
            get_memory_peak_prediction().memory_fragment = fragment
        if not get_adaptive_recomputing_policy().is_find_target_device_memory:
            get_memory_peak_prediction().check_forward_mem_info(state, layer_name, memory_info, is_pre_hook=False)

    def forward_pre_hook(self, prefix, name, ctx):
        def hook(module, *args, **kargs):
            if 'module' in self.context:
                self.context['module'].append(ctx)
            self.pre_hook_func(ctx, prefix, name, *args, **kargs)

        return hook

    def forward_post_hook(self, prefix, name, ctx):
        def hook(module, args, output):
            self.post_hook_func(ctx, prefix, name, args, output)
            if 'module' in self.context:
                self.context['module'].pop()

        return hook

    def construct_context_recursive(self, prefix_name, model, ctx, have_allowed_recomputing):
        # 1.construct context
        next_have_allowed_recomputing = have_allowed_recomputing
        for name, module in model.named_children():
            if 'layers' not in ctx:
                ctx['layers'] = []

            current_ctx = {'name': name, 'prefix_name': prefix_name}
            if 'layers' in ctx:
                ctx['layers'].append(current_ctx)

            next_name = prefix_name + "." + name if prefix_name != "" else name

            # 2.tag allowed_recomputing module
            if have_allowed_recomputing:
                for allowed_recomputing_module in self.allowed_recomputing_module:
                    if isinstance(module, allowed_recomputing_module):
                        current_ctx['allowed_recomputing'] = True
                        if isinstance(model, torch.nn.ModuleList):
                            ctx['is_module_list'] = True
                            ctx['is_recomputing_layer'] = True
                        else:
                            current_ctx['is_recomputing_layer'] = True
                        next_have_allowed_recomputing = False
            self.construct_context_recursive(next_name, module, current_ctx, next_have_allowed_recomputing)

    def register_recursive_hook(self, model, ctx, profiling_prefix):
        index = 0
        for module in model.children():
            if 'layers' not in ctx:
                continue
            current_ctx = ctx['layers'][index]
            # only has allowed_recomputing Tag can set recomputing hook            
            if 'is_module_list' in ctx and 'allowed_recomputing' in current_ctx and index != 0:
                # transformer layer
                module.no_checkpoint_forward = module.forward
                module.forward = get_recompute_hook().hook_checkpoint_forward(module.forward)
                self.checkpointed_modules.append(module)
            prefix_name = current_ctx['prefix_name']
            name = current_ctx['name']

            # profiling entire module
            if "module" == prefix_name:
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(prefix_name, name, current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(prefix_name, name, current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)

            # profiling transformer Layers
            if isinstance(module, torch.nn.ModuleList) and 'is_recomputing_layer' in current_ctx:
                pre_hook = model.register_forward_pre_hook(self.forward_pre_hook(ctx['prefix_name'], ctx['name'], ctx))
                post_hook = model.register_forward_hook(self.forward_post_hook(ctx['prefix_name'], ctx['name'], ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)
            elif 'is_recomputing_layer' in current_ctx:
                profiling_prefix = prefix_name + "." + name
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(prefix_name, name, current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(prefix_name, name, current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)

            # only has allowed_recomputing Tag and its submodule can set profiling hook
            if ('allowed_recomputing' in current_ctx and index == 0):
                profiling_prefix = prefix_name + "." + name
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(prefix_name, name, current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(prefix_name, name, current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)
            elif profiling_prefix and prefix_name.startswith(profiling_prefix):
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(prefix_name, name, current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(prefix_name, name, current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)
            self.register_recursive_hook(module, current_ctx, profiling_prefix)
            index += 1

    def reset_modules(self):
        for m in self.checkpointed_modules:
            m.forward = m.no_checkpoint_forward
        self.checkpointed_modules.clear()
        get_recompute_hook().reset_recompute_modules()
        get_swap_hook().reset_swap_manager_modules()
        SwapManager().reset_swap_manager_tensors()
        if (get_adaptive_recomputing_policy().check_non_oom_times == 0
                and not get_adaptive_recomputing_policy().is_find_target_device_memory):
            torch_npu.npu.empty_cache()

    def reset_all_hook_args(self):
        all_args = get_args()
        step = get_adaptive_recomputing_policy().remove_swap_manager_hook_step
        if not all_args.adaptive_recompute_device_swap:
            for hook_handle in self.modules_hooks:
                hook_handle.remove()
            self.modules_hooks.clear()
            SwapManager().reset_swap_manager_tensors()
            get_swap_hook().reset_swap_manager_modules()
            return

        if not get_adaptive_recomputing_policy().is_find_target_device_memory or self.profiling_step > step + 1:
            return
        get_memory_peak_prediction().memory_fragment = 0
        if self.profiling_step == step + 1:
            title = (f"===== finish to check policy, search policy memory size is: "
                     f"{get_adaptive_recomputing_policy().cur_device_memory} =====")
            print_rank_0(f"{title}\n{get_graph_solver().final_policy_info}\n{'=' * len(title)}")
        if self.profiling_step == step:
            get_swap_hook().reset_swap_manager_modules()
        if get_adaptive_recomputing_policy().is_find_target_device_memory:
            for hook_handle in self.modules_hooks:
                hook_handle.remove()
            self.modules_hooks.clear()
            SwapManager().reset_swap_manager_tensors()

    def step_hook(self, models):
        torch.npu.synchronize()
        self.reset_all_hook_args()
        if (self.profiling_step < self.solve_graph_at_step
                or (self.profiling_step > self.solve_graph_at_step and get_adaptive_recomputing_policy().is_stable_policy(
                    self.profiling_step))):
            return

        if get_adaptive_recomputing_policy().context_copy is None:
            get_adaptive_recomputing_policy().context_copy = deepcopy(self.context)
            try:
                get_adaptive_recomputing_policy().get_default_device_memory(self.context["max_device_memory"])
            except KeyError:
                print_rank_0("[ERROR] Some of these keys don't exist.")
            get_graph_solver().build_solver_info(self.context, self.num_warmup_micro_batches)

        get_adaptive_recomputing_policy().check_cur_recompute_policy()
        print_rank_0("ADAPTIVE-RECOMPUTE: solving recompute policy")
        print_rank_0("==================== ADAPTIVE-RECOMPUTE Report ====================")
        context = get_adaptive_recomputing_policy().solve_recompute_policy(self.profiling_step)
        print_rank_0("==================== ADAPTIVE-RECOMPUTE Report End ====================")
        if context is not None:
            self.context = context
            self.reset_modules()
            print_rank_0("ADAPTIVE-RECOMPUTE: applying policy to the model")
            config = {
                "pre_layer_full_name": "",
                "pre_layer_ctx": {},
                "cur_layer_name": "module",
            }
            apply_adaptive_recompute(config, models, self.context)
            print_rank_0("ADAPTIVE-RECOMPUTE: applying policy to the model fin")
        get_swap_hook().reset_tensor_layer_info()

    def hook_step_func(self, step_func, models):
        def custom_step_func(*args, **kargs):
            result = step_func(*args, **kargs)
            memory_info = self.get_memory_status()
            try:
                hccl_memory = (memory_info["all_memory"] - memory_info["free"] - memory_info[
                    "reserved_memory"]) // self.unit_mb
                get_adaptive_recomputing_policy().hccl_memory = max(hccl_memory, get_adaptive_recomputing_policy().hccl_memory)
                self.context['used_mem'] = memory_info["used_memory"] // self.unit_mb
                self.context['max_device_memory'] = memory_info["all_memory"] // self.unit_mb
            except KeyError:
                print_rank_0("[ERROR] Some of these keys don't exist.")
            self.profiling_step += 1
            self.step_hook(models)
            return result

        return custom_step_func

    def set_profiling_step(self, step):
        self.stop_profiling_step = step
        self.solve_graph_at_step = step + 1

    def add_allowed_recomputing_module(self, module):
        if module not in self.allowed_recomputing_module:
            self.allowed_recomputing_module.append(module)

    def _cal_input_output_size(self, args):
        size = 0
        if isinstance(args, torch.Tensor):
            size += get_tensor_mem_size(args)
            return size // self.unit_mb
        for arg in args:
            if isinstance(arg, torch.Tensor):
                size += get_tensor_mem_size(arg)
            elif isinstance(arg, Iterable):
                for t in arg:
                    if isinstance(t, torch.Tensor):
                        size += get_tensor_mem_size(t)
                    elif t is None:
                        pass
                    else:
                        print_rank_0(f"[WARNING]: unknown input/output type {str(type(t))}")
            elif arg is None:
                pass
            else:
                print_rank_0(f"[WARNING]: unknown input/output type {str(type(t))}")
        return size // self.unit_mb


def get_adaptive_recomputing():
    if AdaptiveRecompute.adaptive_recomputing is None:
        AdaptiveRecompute.adaptive_recomputing = AdaptiveRecompute()
    return AdaptiveRecompute.adaptive_recomputing


def is_activate_adaptive_recompute():
    all_args = get_args()
    profiling_step = 0
    if all_args.adaptive_recompute_device_size < 0 and not all_args.adaptive_recompute_device_swap:
        print_rank_0("[ERROR] failed to activate adaptive selective recompute train, please add param: "
                     "\"adaptive-recompute-device-swap\", or set param: \"adaptive-recompute-device-size\".")
        return False, profiling_step
    if all_args.recompute_granularity is not None or all_args.recompute_method is not None:
        print_rank_0("[ERROR] failed to activate adaptive selective recompute train, please check whether the "
                     "recomputing args switch is turned on: \"recompute_granularity\", \"recompute_method\".")
        return False, profiling_step
    max_profiling_step = all_args.train_iters // 10
    profiling_step = all_args.adaptive_recompute_profiling_step
    if profiling_step < 5 or profiling_step > max_profiling_step:
        print_rank_0(f"[WARNING] consider set \"adaptive-recompute-profiling-step\" value >=5"
                     f"and <={max_profiling_step}, or remove it.")
    if profiling_step <= 0:
        print_rank_0("[WARNING] \"adaptive-recompute-profiling-step\" value can not <=0, will use default value 10.")
        profiling_step = 10
    print_rank_0(
        "success to activate adaptive recompute train: adaptive-recompute-device-swap={}, adaptive-recompute-device-size={}, "
        "adaptive-recompute-profiling-step={}".format(all_args.adaptive_recompute_device_swap,
                                                  all_args.adaptive_recompute_device_size, profiling_step))
    return True, profiling_step


def setup_model_and_optimizer_decorator(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(*args, **kargs):
        models, optimizer, opt_param_scheduler = setup_model_and_optimizer(*args, **kargs)
        activated, profile_step = is_activate_adaptive_recompute()
        if not activated:
            return models, optimizer, opt_param_scheduler
        recomputing = get_adaptive_recomputing()
        recomputing.set_profiling_step(profile_step)
        recomputing.get_num_warmup_micro_batches(len(models))
        optimizer.step = recomputing.hook_step_func(optimizer.step, models)
        if isinstance(models, list):
            for model in models:
                recomputing.construct_context_recursive("module", model, recomputing.context, True)
                recomputing.register_recursive_hook(model, recomputing.context, recomputing.profiling_prefix)
        else:
            recomputing.construct_context_recursive("module", models, recomputing.context, True)
            recomputing.register_recursive_hook(models, recomputing.context, recomputing.profiling_prefix)
        print_rank_0("ADAPTIVE-RECOMPUTE: successfully hooking module")
        return models, optimizer, opt_param_scheduler

    return wrapper


def allowed_recomputing_module_wrapper(allowed_recomputing_module):
    recomputing = get_adaptive_recomputing()
    recomputing.add_allowed_recomputing_module(allowed_recomputing_module)