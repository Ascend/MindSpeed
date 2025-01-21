# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import sys
import time
import pickle
from copy import deepcopy
from typing import List

import torch
import numpy as np
from megatron.training import print_rank_0, get_args
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core import parallel_state as ps
from .adaptive_memory_cache import AdaptiveModelMemPolicy, PolicyCacheManager
from .adaptive_memory_policy import AdaptMemPolicyManager
from .adaptive_memory_swap_manager import SwapManager
from .adaptive_memory_tool import SingletonBase, LayerAction, ModuleAction, FuncLocation, ContextKey as Key, BYTES_PER_MB
from .adaptive_memory_tool import AdaptiveStepMgr, broadcast_obj


class AdaptMemGraphSolver(metaclass=SingletonBase):
    def __init__(self):
        self.num_warmup_bs_in_chunks = self.get_chunk_num_warmup_micro_batches()
        self.adapt_mem_policy = {}
        self.static_memory = 0
        self.best_layer_policy_comb = []
        self.func_locations: List[FuncLocation] = []
        self.need_prepare_solver = True

        self.device_memory = sys.maxsize
        self.cur_adapt_policy = None
        self.swap_size = 0
        self.record_swap_out_size = 0
        self.last_num_alloc_retries = torch.npu.memory_stats()["num_alloc_retries"]
        self.remove_swap_manager_hook_step = 0
        self.cur_device_memory = -1
        self.flag_find_target_memory = False
        self.first_non_oom_device_memory = 0
        self.min_dichotomy_value = 1
        self.dichotomy_memory_left = 0
        self.dichotomy_memory_right = 0
        self.alloc_retries_times = 0 # 记录当前策略alloc失败的次数
        self.is_stable_for_non_oom_policy = 1 # 判断非oom的策略是否稳定==>1：稳定、0：不稳定

    def prepare_solver(self, model_context):
        self.need_prepare_solver = False
        self.static_memory = self.get_static_mem(model_context)
        self.dichotomy_memory_left = self.static_memory

    @staticmethod
    def get_chunk_num_warmup_micro_batches():
        num_warmup_bs_in_chunks = []
        pp = ps.get_pipeline_model_parallel_world_size()
        vpp = ps.get_virtual_pipeline_model_parallel_world_size() or 1
        pp_rank = ps.get_pipeline_model_parallel_rank()
        num_micro_batches = get_num_microbatches()
        if pp <= 1 or None in (num_micro_batches, pp_rank, vpp):
            return [1]
        elif vpp == 1:
            num_warmup_bs = pp - pp_rank - 1
            num_warmup_bs += 1
            num_warmup_bs_in_chunks.append(num_warmup_bs)
        else:
            total_num_micro_batches = num_micro_batches * vpp
            num_warmup_bs = (pp - pp_rank - 1) * 2
            num_warmup_bs += (vpp - 1) * pp
            num_warmup_bs += 1
            num_warmup_bs = min(num_warmup_bs, total_num_micro_batches)
            remain_batch_num = (num_warmup_bs - pp * vpp)
            for i in range(vpp):
                if i == 0:
                    num_warmup_bs_in_chunks.append(pp + max(0, remain_batch_num))
                elif i == vpp - 1:
                    num_warmup_bs_in_chunks.append(pp + min(0, remain_batch_num))
                else:
                    num_warmup_bs_in_chunks.append(pp)

        print_rank_0(f"layer_num:{get_args().num_layers}")
        print_rank_0(f"pp:{pp}")
        print_rank_0(f"vpp:{vpp}")
        print_rank_0(f"pp_rank:{pp_rank}")
        print_rank_0(f"num_micro_batches:{num_micro_batches}")
        print_rank_0(f"num_warmup_bs_in_chunks:{num_warmup_bs_in_chunks}")
        print_rank_0(f"layer_num_per_ppstage:{get_args().num_layers // ps.get_pipeline_model_parallel_world_size()}")

        return num_warmup_bs_in_chunks

    @staticmethod
    def tensor_all_reduce(num_list, op):
        # all reduce the "num_list" between tp ranks in group
        reduce_tensor = torch.tensor(num_list, device=torch.npu.current_device())
        if ps.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(reduce_tensor, op=op, group=ps.get_tensor_model_parallel_group())
        # all reduce the "num_list" between dp ranks in group
        if ps.get_data_parallel_world_size(True) > 1:
            torch.distributed.all_reduce(reduce_tensor, op=op, group=ps.get_data_parallel_group(True))
        result = reduce_tensor.cpu().numpy().tolist()
        del reduce_tensor
        return result

    def is_stable_policy(self):
        if AdaptiveStepMgr().get_cur_step() > self.remove_swap_manager_hook_step != 0:
            return True
        total_swap_out_size = SwapManager().oom_rescue_total_swap_out_size
        self.swap_size = (total_swap_out_size - self.record_swap_out_size) // BYTES_PER_MB
        self.check_num_alloc_retries()
        num_list = [
            int(total_swap_out_size), int(AdaptMemPolicyManager().hccl_memory), int(self.swap_size),
            int(self.flag_find_target_memory), int(self.alloc_retries_times)
        ]
        size_tensor = self.tensor_all_reduce(num_list, torch.distributed.ReduceOp.MAX)
        total_swap_out_size = size_tensor[0]
        AdaptMemPolicyManager().hccl_memory = size_tensor[1]
        self.swap_size = size_tensor[2]
        self.flag_find_target_memory = bool(size_tensor[3])
        self.alloc_retries_times = size_tensor[4]
        SwapManager().oom_rescue_total_swap_out_size = total_swap_out_size

        if self.swap_size <= 0 and self.flag_find_target_memory:
            return True
        self.record_swap_out_size = total_swap_out_size
        return False

    def check_num_alloc_retries(self):
        num_alloc_retries = torch.npu.memory_stats()["num_alloc_retries"]
        # if policy is normal and stable
        if num_alloc_retries == self.last_num_alloc_retries:
            return
        retries_times = num_alloc_retries - self.last_num_alloc_retries
        self.last_num_alloc_retries = num_alloc_retries
        # policy tag oom if policy is unstable
        if self.swap_size == 0 and (retries_times > 1 or self.is_stable_for_non_oom_policy == 0):
            self.swap_size = 1
        # if policy is oom or unstable
        if self.swap_size > 0:
            return

        self.alloc_retries_times += 1
        if self.alloc_retries_times > 1:
            print_rank_0("this is a unstable policy, try select another one.")
            self.swap_size = 1

    def reduce_device_memory(self, device_memory):
        cur_min_memory = min(self.device_memory, device_memory)
        self.device_memory, = self.tensor_all_reduce([int(cur_min_memory)], torch.distributed.ReduceOp.MIN)
        print_rank_0(f"reduce device memory from {device_memory} to {self.device_memory}")

    def check_cur_adapt_policy(self):
        if not self.cur_adapt_policy:
            return

        policy_cache_manager = PolicyCacheManager()
        flag_in_oom_list = policy_cache_manager.check_in_oom_cache(self.cur_adapt_policy)
        flag_in_normal_list = policy_cache_manager.check_in_normal_cache(self.cur_adapt_policy)
        if self.swap_size > 0:
            if not flag_in_oom_list:
                policy_cache_manager.add_oom_policy_cache(deepcopy(self.cur_adapt_policy))
            if flag_in_normal_list:
                policy_cache_manager.delete_normal_policy_cache(self.cur_adapt_policy)
            return
        if flag_in_oom_list or self.alloc_retries_times != 0:
            return
        if not flag_in_normal_list:
            policy_cache_manager.add_normal_policy_cache(deepcopy(self.cur_adapt_policy))

    def solve_adapt_mem_policy(self):
        flag_is_known_policy = True
        cur_step = AdaptiveStepMgr().get_cur_step()
        self.remove_swap_manager_hook_step = cur_step + 1
        adapt_policy_list = None
        while flag_is_known_policy:
            torch.npu.synchronize()
            self.cur_device_memory = self.dichotomy_find_memory()
            print_rank_0(f"cur_device_memory:{self.cur_device_memory}")
            if self.is_stable_for_non_oom_policy != 0:  # 对于不稳定的策略不生成新策略，使用旧策略再测试一遍
                adapt_policy_list = self.get_mem_policy(self.cur_device_memory)
                self.cur_adapt_policy = AdaptiveModelMemPolicy("normal", self.best_layer_policy_comb)
            if self.flag_find_target_memory:
                self.remove_swap_manager_hook_step = cur_step + 10
                print_rank_0(
                    f"success to find the target value of the current round of search: {self.cur_device_memory}")
                break
            # OOM policy
            policy_cache_manager = PolicyCacheManager()
            if policy_cache_manager.check_in_oom_cache(self.cur_adapt_policy):
                self.swap_size = max(self.swap_size, 1)
                continue
            # no OOM policy
            if policy_cache_manager.check_in_normal_cache(self.cur_adapt_policy):
                self.swap_size = 0
                continue
            flag_is_known_policy = False

        return adapt_policy_list

    def get_dichotomy_value(self):
        return (self.dichotomy_memory_left + self.dichotomy_memory_right) // 2

    def dichotomy_find_memory(self):
        # last policy is instability
        if self.flag_find_target_memory:
            self.dichotomy_memory_left = self.first_non_oom_device_memory
            self.dichotomy_memory_right = self.cur_device_memory
        self.flag_find_target_memory = False
        if self.cur_device_memory == -1:
            return self.device_memory

        # OOM
        if self.swap_size > 0:
            print_rank_0(f"current policy is OOM, policy device memory: {self.cur_device_memory}")
            self.is_stable_for_non_oom_policy = 1
            self.alloc_retries_times = 0
            self.dichotomy_memory_right = self.cur_device_memory
            if self.first_non_oom_device_memory >= self.cur_device_memory:
                self.first_non_oom_device_memory = 0
            if self.dichotomy_memory_right <= self.static_memory:
                raise ValueError("out of Memory!!!!!!!!!!")
            elif self.dichotomy_memory_right <= self.dichotomy_memory_left:
                self.dichotomy_memory_left = self.static_memory
            return self.get_dichotomy_value()

        # check non oom policy
        if self.alloc_retries_times != 0 and self.is_stable_for_non_oom_policy == 1:
            print_rank_0(f"current policy may be an unstable, policy device memory: {self.cur_device_memory}")
            self.is_stable_for_non_oom_policy = 0
            self.alloc_retries_times = 0
            return self.cur_device_memory

        self.is_stable_for_non_oom_policy = 1
        self.alloc_retries_times = 0
        self.dichotomy_memory_left = self.cur_device_memory
        if self.first_non_oom_device_memory == 0:
            self.first_non_oom_device_memory = self.cur_device_memory
        if self.dichotomy_memory_right - self.dichotomy_memory_left <= self.min_dichotomy_value:
            self.flag_find_target_memory = True
            return self.dichotomy_memory_left

        return self.get_dichotomy_value()

    @staticmethod
    def get_pp_layer_num():
        return get_args().num_layers // ps.get_pipeline_model_parallel_world_size()

    @staticmethod
    def get_layer_num_per_chunk():
        vpp = ps.get_virtual_pipeline_model_parallel_world_size() or 1
        return AdaptMemGraphSolver.get_pp_layer_num() // vpp

    def get_static_mem(self, model_context):
        single_chunk_memory = 0
        num_of_chunk = len(model_context[Key.SUBMODULES])
        if num_of_chunk > 0 and Key.MEMORY in model_context[Key.SUBMODULES][0]:
            single_chunk_memory = model_context[Key.SUBMODULES][0][Key.MEMORY]
        # 不能被节省的动态内存
        mem_space_cannot_be_saved = (single_chunk_memory - AdaptMemPolicyManager().total_adapt_memory) * num_of_chunk
        # 静态内存 = 模型总内存 + 不能被节省的动态内存
        static_mem_size = model_context[Key.USED_MEM] + mem_space_cannot_be_saved
        print_rank_0(f"static_memory:{static_mem_size}")
        return static_mem_size

    def get_mem_policy(self, device_memory):
        print_rank_0("Using the knapsack algorithm to find the optimal strategy")
        self.adapt_mem_policy.clear()
        self.knapsack_best(device_memory)
        adapt_mem_policy_list = self.get_adapt_mem_policy_list()
        print_rank_0(f"adapt_mem_policy_list:{adapt_mem_policy_list}")
        if torch.distributed.is_initialized():
            # 把self.recompute_policy字典转换为recompute_policy_list列表，方便广播到其他卡上
            adapt_mem_policy_list = broadcast_obj(adapt_mem_policy_list)
            self.best_layer_policy_comb = broadcast_obj(self.best_layer_policy_comb)
        return adapt_mem_policy_list

    def get_max_goods_value(self, idx, ans, device_memory):
        i, j, k = idx
        pre_step_ans = ans[i - 1][j - k]
        if k == 0:
            return deepcopy(pre_step_ans)

        goods_value = ans[i][j]
        # calculate memory
        memory = pre_step_ans.memory
        pre_layer_num = len(pre_step_ans.polices)
        for index in range(k):
            cur_layer_index = pre_layer_num + index
            cur_layer_chunk_rank = cur_layer_index // self.get_layer_num_per_chunk()
            cur_layer_bs = self.num_warmup_bs_in_chunks[cur_layer_chunk_rank]
            cur_layer_memory_cost = cur_layer_bs * AdaptMemPolicyManager().policy_combinations[i].memory
            memory += cur_layer_memory_cost
        # calculate cost
        comb_time = pre_step_ans.time + k * AdaptMemPolicyManager().policy_combinations[i].time
        # calculate device_memory
        if pre_step_ans.time == sys.maxsize:
            comb_time = k * AdaptMemPolicyManager().policy_combinations[i].time
        max_free_memory = max(device_memory - self.static_memory, 0)

        if max_free_memory >= memory and comb_time <= goods_value.time and (len(pre_step_ans.polices) + k) == j:
            goods_value.memory = memory
            goods_value.time = comb_time
            goods_value.polices.clear()
            goods_value.polices.extend(pre_step_ans.polices)
            goods_value.polices.extend(AdaptMemPolicyManager().policy_combinations[i] for _ in range(k))

        return goods_value

    def add_func_locations(self, layer_idx, func_name, action):
        self.func_locations.append(FuncLocation(layer_idx, func_name, action))

    def get_cur_layer_idx(self, count):
        pp = ps.get_pipeline_model_parallel_world_size()
        vpp = ps.get_virtual_pipeline_model_parallel_world_size() or 1
        total_layers = get_args().num_layers
        if vpp > 1:
            layers_per_chunk = total_layers // pp // vpp

            # calc count belong to chunk and layer idx
            remain = count % (pp * vpp * layers_per_chunk)
            cur_chunk_idx = remain // (pp * layers_per_chunk)  # 当前chunk id
            cur_layer_idx = remain % (pp * layers_per_chunk) % layers_per_chunk  # 当前layer在chunk内的id
            global_layer_idx = cur_chunk_idx * layers_per_chunk + cur_layer_idx
            return global_layer_idx
        elif pp > 1:
            layers_per_pp = total_layers // pp
            global_layer_idx = count % layers_per_pp
            return global_layer_idx
        else:
            global_layer_idx = count % total_layers
            return global_layer_idx

    def get_func_action(self, function_name, count) -> ModuleAction:
        pp = ps.get_pipeline_model_parallel_world_size()
        total_layers = get_args().num_layers
        layers_per_pp = total_layers // pp

        all_same_func_loc = [x for x in self.func_locations if x.func_name == function_name]
        if len(all_same_func_loc) != layers_per_pp:
            raise AssertionError("get_func_action error.")
        global_layer_idx = self.get_cur_layer_idx(count)
        if global_layer_idx != all_same_func_loc[global_layer_idx].layer_idx:
            raise AssertionError("get_func_action error.")
        return all_same_func_loc[global_layer_idx].action

    def get_mem_layer_policy(self, combination_num, layer_num, ans):
        apm = AdaptMemPolicyManager()
        layer_full_recompute_memory = 0
        for index in range(layer_num):
            cur_layer_index = index
            cur_layer_chunk_rank = cur_layer_index // self.get_layer_num_per_chunk()
            cur_layer_memory_cost = self.num_warmup_bs_in_chunks[cur_layer_chunk_rank] * apm.full_recompute_comb.memory
            layer_full_recompute_memory += cur_layer_memory_cost

        layer_full_recompute_time = layer_num * apm.full_recompute_comb.time

        self.best_layer_policy_comb = [apm.full_recompute_comb for _ in range(layer_num)]

        size = layer_num - len(ans[combination_num][layer_num].polices)
        pre_layer_num = len(ans[combination_num][layer_num].polices)
        memory = ans[combination_num][layer_num].memory
        for index in range(size):
            cur_layer_index = pre_layer_num + index
            cur_layer_chunk_rank = cur_layer_index // self.get_layer_num_per_chunk()
            memory += self.num_warmup_bs_in_chunks[cur_layer_chunk_rank] * apm.full_recompute_comb.memory
        comb_time = ans[combination_num][layer_num].time + size * apm.full_recompute_comb.time
        best_policy_comb = deepcopy(ans[combination_num][layer_num].polices)
        best_policy_comb.extend(size * [apm.full_recompute_comb])

        if comb_time < layer_full_recompute_time:
            self.best_layer_policy_comb.clear()
            self.best_layer_policy_comb = best_policy_comb

        print_rank_0(f"full_recompute_comb.time:{apm.full_recompute_comb.time}")
        print_rank_0(f"full_recompute_comb.memory:{apm.full_recompute_comb.memory}")
        print_rank_0(f"without_adaptive_comb.time:{apm.without_adaptive_comb.time}")
        print_rank_0(f"without_adaptive_comb.memory:{apm.without_adaptive_comb.memory}")
        print_rank_0(f"full_swap_comb.time:{apm.full_swap_comb.time}")
        print_rank_0(f"full_swap_comb.memory:{apm.full_swap_comb.memory}")

        for policy in self.best_layer_policy_comb:
            policy_recompute = str(policy.recompute)
            policy_swap = str(policy.swap)
            if (policy_recompute, policy_swap) in self.adapt_mem_policy.keys():
                self.adapt_mem_policy[policy_recompute, policy_swap] += 1
            else:
                self.adapt_mem_policy[policy_recompute, policy_swap] = 1
        print_rank_0(f"adapt_mem_policy_dict:{self.adapt_mem_policy}")

    def knapsack_best(self, device_memory):
        start_time = time.time()
        combination_num = len(AdaptMemPolicyManager().policy_combinations) - 1
        if AdaptMemPolicyManager().policy_combinations[0] is not None:
            combination_num = len(AdaptMemPolicyManager().policy_combinations)
            # make combination index id begin for 1.
            AdaptMemPolicyManager().policy_combinations.insert(0, None)
        print_rank_0(f"combination_num:{combination_num}")

        # init ans
        def default_policy():
            return AdaptiveModelMemPolicy("normal", [])

        ans = [[default_policy() for _ in range(self.get_pp_layer_num() + 1)] for _ in range(combination_num + 1)]

        # find max goods value
        for i in range(1, combination_num + 1):
            for j in range(self.get_pp_layer_num() + 1):
                if i >= 2:
                    ans[i - 2][j].polices.clear()
                for k in range(j + 1):
                    ans[i][j] = self.get_max_goods_value([i, j, k], ans, device_memory)
        self.get_mem_layer_policy(combination_num, self.get_pp_layer_num(), ans)
        end_time = time.time()
        execution_time = end_time - start_time
        print_rank_0(f"The execution time of the knapsack algorithm is {execution_time} seconds.")

    def get_adapt_mem_policy_list(self):
        adapt_mem_policy_list = []
        apm = AdaptMemPolicyManager()
        for key, times in self.adapt_mem_policy.items():
            temp_adapt_mem_policy_list = [times]
            key_recompute = eval(key[0])
            key_swap = eval(key[1])
            if key_recompute == apm.without_adaptive_comb.recompute and key_swap == apm.without_adaptive_comb.swap:
                temp_adapt_mem_policy_list.append(LayerAction.NONE)
                temp_adapt_mem_policy_list.extend([ModuleAction.NONE] * apm.adapt_modules_num)
            elif key_recompute == apm.full_recompute_comb.recompute and key_swap == apm.full_recompute_comb.swap:
                temp_adapt_mem_policy_list.append(LayerAction.FULL_RECOMPUTE)
                temp_adapt_mem_policy_list.extend([ModuleAction.RECOMPUTE] * apm.adapt_modules_num)
            elif key_recompute == apm.full_swap_comb.recompute and key_swap == apm.full_swap_comb.swap:
                temp_adapt_mem_policy_list.append(LayerAction.FULL_SWAP)
                temp_adapt_mem_policy_list.extend([ModuleAction.SWAP] * apm.adapt_modules_num)
            else:
                temp_adapt_mem_policy_list.append(LayerAction.ADAPTIVE)
                for module_name in apm.module_layers_name:
                    if module_name in key_recompute:
                        temp_adapt_mem_policy_list.append(ModuleAction.RECOMPUTE)
                    elif module_name in key_swap:
                        temp_adapt_mem_policy_list.append(ModuleAction.SWAP)
                    else:
                        temp_adapt_mem_policy_list.append(ModuleAction.NONE)
            adapt_mem_policy_list.append(temp_adapt_mem_policy_list)
        return adapt_mem_policy_list
