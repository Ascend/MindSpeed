import sys
from copy import deepcopy

import acl
import torch
from megatron.training import print_rank_0

from .adaptive_memory_cache import AdaptiveLayerMemPolicy
from .adaptive_memory_prefetch import AdaptiveMemoryPrefetch
from .adaptive_memory_tool import AdaptiveStepMgr, SingletonBase, ModuleAction, LayerAction, ContextKey as Key


class AdaptMemPolicyManager(metaclass=SingletonBase):

    def __init__(self):
        self.hccl_memory = 0

        # policy combinations
        self.policy_combinations = []
        self.without_adapt_mem = 0.0
        self.full_recompute_comb = None
        self.full_swap_comb = None
        self.without_adaptive_comb = None
        # solve policy
        self.adapt_modules_num = 0
        self.total_adapt_memory = 0.0
        self.module_layers_name = []
        # adaptive prefetch
        self.prefetch_parents_comb = []
        self.memory_interval = 1

    def prepare_policy(self, model_context):
        self.traversal_model_context(model_context)
        for comb in self.policy_combinations:
            comb.memory = comb.memory + self.without_adapt_mem
        # select policy that contains prefetch_parents_comb
        self.select_policy_with_prefetch_parents_comb()

    def traversal_model_context(self, context):
        for layer_context in context.get(Key.SUBMODULES, []):
            # 统计一下做自适应的总动态内存
            if Key.IS_ADAPT_LAYER in layer_context and Key.MEMORY in context:
                self.total_adapt_memory += context[Key.MEMORY]
            if Key.ALLOWED_ADAPT in layer_context and Key.MEMORY in layer_context:
                self.generate_full_combinations(layer_context, self.policy_combinations, "", 0, False)
                return
            else:
                self.traversal_model_context(layer_context)

    def generate_full_combinations(self, ctx, pre_policy_comb, pre_allow_adapt_n, idx, without_layer):
        new_policy_comb = []
        cycle_policy_comb = pre_policy_comb.copy()
        if pre_allow_adapt_n:
            ctx[Key.PREFIX_NAME] = remove_content_before_target(ctx[Key.PREFIX_NAME], pre_allow_adapt_n)
        # root layers
        if idx == 0:
            self.build_initial_combinations(ctx)
            pre_allow_adapt_n = ctx[Key.PREFIX_NAME] + '.'
            self.prefetch_parents_comb = self.generate_prefetch_policy_combinations(pre_allow_adapt_n)
        elif Key.MEMORY in ctx:
            self.adapt_modules_num += 1
            self.module_layers_name.append(ctx[Key.PREFIX_NAME] + "." + ctx[Key.NAME])
            new_policy_comb.extend(self.build_combinations(ctx, pre_policy_comb, ModuleAction.SWAP, without_layer))
            if ctx[Key.MEMORY] > ctx[Key.INPUT] + ctx[Key.OUTPUT] + self.memory_interval and not ctx[Key.IS_SWAP]:
                new_policy_comb.extend(self.build_combinations(ctx, pre_policy_comb, ModuleAction.RECOMPUTE, without_layer))
        if check_all_sub_same_mem(ctx):
            same_mep_comb = pre_policy_comb.copy()
            for sub in ctx.get(Key.SUBMODULES, []):
                same_mep_comb = self.generate_full_combinations(sub, same_mep_comb, pre_allow_adapt_n, idx + 1, True)
                new_policy_comb.extend(same_mep_comb)
            cycle_policy_comb.extend(same_mep_comb)
        else:
            for sub in ctx.get(Key.SUBMODULES, []):
                tmp_combs = self.generate_full_combinations(sub, cycle_policy_comb, pre_allow_adapt_n, idx + 1, False)
                cycle_policy_comb.extend(tmp_combs)
                new_policy_comb.extend(tmp_combs)
        return new_policy_comb

    def build_initial_combinations(self, context):
        self.without_adapt_mem = context[Key.MEMORY]
        self.full_recompute_comb = AdaptiveLayerMemPolicy(recompute=[context[Key.NAME]], swap=[],
                                                          memory=context[Key.INPUT] + context[Key.OUTPUT] - self.without_adapt_mem,
                                                          time=context[Key.AVG_TIME],
                                                          adapt_type=LayerAction.FULL_RECOMPUTE)
        self.full_swap_comb = AdaptiveLayerMemPolicy(recompute=[], swap=[context[Key.NAME]],
                                                     memory=-context[Key.MODULE_SWAP_AVG_MEMORY],
                                                     time=context[Key.MODULE_SWAP_AVG_TIME], adapt_type=LayerAction.FULL_SWAP)
        self.without_adaptive_comb = AdaptiveLayerMemPolicy(recompute=[], swap=[],
                                                            memory=0, time=0,
                                                            adapt_type=LayerAction.NONE)
        self.policy_combinations.append(self.full_recompute_comb)
        self.policy_combinations.append(self.without_adaptive_comb)

    def generate_prefetch_policy_combinations(self, pre_allow_adapt_n):
        prefetch_policy = AdaptiveLayerMemPolicy(time=0)
        for module_name in AdaptiveMemoryPrefetch().need_swap_module_name:
            suffix_name = remove_content_before_target(module_name, pre_allow_adapt_n)
            prefetch_policy.swap.append(suffix_name)
        return prefetch_policy


    def build_combinations(self, context, pre_policy_combs, adapter_tag, without_cur_layer):
        new_policy_combs = []
        cur_policy_combs = pre_policy_combs.copy()
        for policy_comb in cur_policy_combs:
            new_policy_combs.append(self.build_one_combination(context, policy_comb, adapter_tag))
        if without_cur_layer:
            return new_policy_combs
        single_policy_comb = self.build_one_combination(context, AdaptiveLayerMemPolicy(time=0), adapter_tag)
        new_policy_combs.append(single_policy_comb)
        return new_policy_combs

    def build_one_combination(self, context, pre_policy_comb, adapter_tag):
        layer_name = context[Key.PREFIX_NAME] + '.' + context[Key.NAME]
        layer_list = pre_policy_comb.get_modules_by_tag(adapter_tag).copy()
        policy_comb = AdaptiveLayerMemPolicy()
        layer_list.append(layer_name)
        if ModuleAction.RECOMPUTE == adapter_tag:
            policy_comb.swap = pre_policy_comb.swap.copy()
            policy_comb.recompute = layer_list
            policy_comb.memory = pre_policy_comb.memory - context[Key.MEMORY] + context[Key.INPUT] + context[Key.OUTPUT]
            policy_comb.time = pre_policy_comb.time + context[Key.AVG_TIME]
        if ModuleAction.SWAP == adapter_tag:
            policy_comb.recompute = pre_policy_comb.recompute.copy()
            policy_comb.swap = layer_list
            # if the module has swap information
            if Key.MODULE_SWAP_AVG_MEMORY in context:
                policy_comb.memory = pre_policy_comb.memory - context[Key.MODULE_SWAP_AVG_MEMORY]
                if context[Key.IS_SWAP]:
                    # if swap doesn't waste time
                    policy_comb.time = pre_policy_comb.time
                else:
                    policy_comb.time = pre_policy_comb.time + context[Key.MODULE_SWAP_AVG_TIME]
            else:
                policy_comb.memory = pre_policy_comb.memory
                policy_comb.time = pre_policy_comb.time
        self.policy_combinations.append(policy_comb)
        return policy_comb

    def select_policy_with_prefetch_parents_comb(self):
        new_policy_comb = []
        for policy_comb in self.policy_combinations:
            if policy_comb.adapt_type != LayerAction.ADAPTIVE:
                new_policy_comb.append(policy_comb)
            elif self.is_contained_prefetch_parents_comb(self.prefetch_parents_comb.swap, policy_comb.swap):
                new_policy_comb.append(policy_comb)
        self.policy_combinations = new_policy_comb

    def is_contained_prefetch_parents_comb(self, prefetch_parents_list, swap_list):
        prefetch_parents_list_copy = prefetch_parents_list.copy()
        prefetch_parents_list_copy.sort()
        swap_list_copy = swap_list.copy()
        swap_list_copy.sort()
        return prefetch_parents_list_copy == swap_list_copy


    def update_hccl_memory(self):
        free, all_memory, _ = acl.rt.get_mem_info(1)
        cur_hccl_memory = (all_memory - free - torch.npu.memory_reserved()) / 1024 / 1024
        self.hccl_memory = max(cur_hccl_memory, self.hccl_memory)

    def is_stable_mem_policy(self):
        if not AdaptiveStepMgr().is_all_profiling_done():
            return False
        if not AdaptiveMemoryPrefetch().is_stable_apply:
            return False
        from .adaptive_memory_solver import AdaptMemGraphSolver
        if not AdaptMemGraphSolver().is_stable_policy():
            return False
        return True


def remove_content_before_target(path: str, prefix: str):
    if path.startswith(prefix):
        return path[len(prefix):]
    else:
        return path


def check_all_sub_same_mem(context):
    submodules = [child for child in context.get(Key.SUBMODULES, []) if Key.MEMORY in child]
    for i in range(len(submodules) - 1):
        if submodules[i][Key.MEMORY] != submodules[i + 1][Key.MEMORY]:
            return False
    return True
