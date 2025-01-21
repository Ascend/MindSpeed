# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from functools import wraps

import acl
import torch
import torch.nn

from megatron.training import print_rank_0
from megatron.core.num_microbatches_calculator import get_num_microbatches
from .adaptive_memory_profiling import AdaptiveMemoryProfiling, RecomputeHook
from .adaptive_memory_solver import AdaptMemGraphSolver
from .adaptive_memory_policy import AdaptMemPolicyManager
from .adaptive_memory_prefetch import AdaptiveMemoryPrefetch
from .adaptive_memory_tool import AdaptiveStepMgr, SingletonBase, ContextKey as Key, ForwardCounter
from .adaptive_memory_function import FunctionCtxMgr
from .adaptive_memory_swap_manager import SwapManager, LayerProfilingHook
from .adaptive_memory_apply import AdaptMemApplyManager
from .cpu_binder import bind_cpus
from .adaptive_memory_cache import PolicyCacheManager


class AdaptiveMemoryOpt(metaclass=SingletonBase):

    @staticmethod
    def reset_all_adapt_mem_hooks():
        if not AdaptiveStepMgr().is_recompute_profiling_step():
            AdaptiveMemoryProfiling().reset_profiling_all_hooks()

        if AdaptiveMemoryOpt.is_policy_stable():
            AdaptiveMemoryOpt.reset_final_rescue_hooks()

    @staticmethod
    def is_policy_stable():
        # current policy run 10 more steps unchanged is a stable policy
        return AdaptiveStepMgr().get_cur_step() >= AdaptMemGraphSolver().remove_swap_manager_hook_step

    @staticmethod
    def reset_final_rescue_hooks():
        SwapManager().reset_oom_rescue_hooked_modules()

    @staticmethod
    def reset_adapt_mem_modules():
        RecomputeHook().reset_recompute_modules() # clear recompute modules
        AdaptiveMemoryProfiling().reset_profiling_all_hooks() # clear profiling all hook
        AdaptiveMemoryPrefetch().reset_adaptive_prefetch_all_hooks() # clear adaptive prefetch all hook
        SwapManager().reset_all_for_oom_rescue() # clear all hook and tensor in oom rescue

    def set_adapt_mem_hook(self, models):
        torch.npu.synchronize()
        AdaptiveMemoryProfiling().record_time()
        context = AdaptiveMemoryProfiling().context
        # reset auto_function list
        if not AdaptiveMemoryPrefetch().is_stable_apply:
            AdaptiveMemoryPrefetch().function_swap_profiling_deep = 0
            AdaptiveMemoryPrefetch().prefetch_function_list = []
            AdaptiveMemoryPrefetch().prefetch_module_dict.clear()

        if AdaptiveStepMgr().is_recompute_profiling_step():
            if AdaptiveStepMgr().is_last_recompute_profiling_step():
                # insert function profiling to context
                for ctx, child in FunctionCtxMgr().ctx_iter():
                    AdaptiveMemoryProfiling().insert_func_profiling(ctx, child)
                # update params when has function
                if len(FunctionCtxMgr()._ctx_dict):
                    update_swap_profiling_step_and_deep_list()

                # clear recompute profiling hook
                AdaptiveMemoryProfiling().reset_profiling_hooks()
                AdaptiveMemoryPrefetch().reset_adaptive_prefetch_all_hooks()
                # apply layer profiling hook for following steps
                LayerProfilingHook().apply_layer_profiling_hook(AdaptiveMemoryProfiling().layer0_module)
            return

        if AdaptiveStepMgr().is_layer_profiling_step():
            if AdaptiveStepMgr().is_last_layer_profiling_step():
                SwapManager().forward_time = LayerProfilingHook().get_single_layer_time()
                LayerProfilingHook().reset_layer_profiling_hook()
                LayerProfilingHook().forward_time_list.clear()
                print_rank_0(f'forward time is {SwapManager().forward_time}')
                config = AdaptiveMemoryPrefetch().config
                AdaptiveMemoryPrefetch().register_recursive_apply_prefetch(config, models, context)
            return

        # update swap profiling stats
        if AdaptiveStepMgr().is_swap_profiling_step():
            AdaptiveMemoryPrefetch().update_ctx(models, context)


        if AdaptiveStepMgr().is_swap_profiling_done() and not AdaptiveMemoryPrefetch().is_stable_apply:
            AdaptiveMemoryPrefetch().adaptive_select_module(models, context)
        if not AdaptiveMemoryPrefetch().is_stable_apply:
            return

        if AdaptMemGraphSolver().need_prepare_solver:
            # reduce max_device_memory and generate all policy combinations at first solver step
            AdaptMemGraphSolver().reduce_device_memory(context[Key.DEVICE_MEMORY])
            AdaptMemPolicyManager().prepare_policy(context)
            AdaptMemGraphSolver().prepare_solver(context)

        AdaptMemGraphSolver().check_cur_adapt_policy()
        print_rank_0("==================== ADAPTIVE-MEMORY Report START====================")
        adapt_policy_list = AdaptMemGraphSolver().solve_adapt_mem_policy()
        print_rank_0("==================== ADAPTIVE-MEMORY Report End ====================")
        if adapt_policy_list is not None:
            self.reset_adapt_mem_modules()
            AdaptMemApplyManager().apply_new_adapt_policy(adapt_policy_list, context, models)
            print_rank_0(f"ADAPTIVE MEMORY OPTIMIZATION apply policy done")

    def hook_adapt_mem_step(self, step_func, models):
        def custom_adapt_mem_step(*args, **kwargs):
            try:
                result = step_func(*args, **kwargs)     # cur step is done after calling step_func
                if AdaptMemPolicyManager().is_stable_mem_policy() or AdaptiveStepMgr().is_skipping_step():
                    return result

                AdaptiveMemoryProfiling().update_whole_model_memory()
                AdaptMemPolicyManager().update_hccl_memory()
                self.set_adapt_mem_hook(models)

                return result
            finally:
                AdaptiveStepMgr().incr_step()           # incr step num after step_func and adapting

        return custom_adapt_mem_step


def addup_allowed_mem_adapt_module(module):
    AdaptiveMemoryProfiling().addup_allowed_mem_adapt_profiling_module(module)


def layer_beginning_callback_forward(module, *args, **kwargs):
    ForwardCounter().incr_cnt()


def register_custom_hooks(modules):
    for module in modules:
        _register_one_module(module)


def _register_one_module(module):
    allowed_list = AdaptiveMemoryProfiling().get_allowed_adapt_module()
    if any(isinstance(module, a) for a in allowed_list):
        module.register_forward_pre_hook(layer_beginning_callback_forward)

    for name, child in module.named_children():
        if isinstance(child, torch.nn.ModuleList):
            for idx, sub_child in enumerate(child):
                _register_one_module(sub_child)
        else:
            _register_one_module(child)


def cal_swap_profiling_step(num_micro_batches):
    swap_depth = AdaptiveMemoryPrefetch().prefetch_deep_end - AdaptiveMemoryPrefetch().prefetch_deep_start + 1
    swap_profiling_times = 4
    swap_profiling_steps = swap_profiling_times // num_micro_batches
    if swap_profiling_times % num_micro_batches != 0:
        swap_profiling_steps += 1
    return swap_profiling_steps * swap_depth * AdaptiveMemoryPrefetch().each_depth_run_times


def cal_profiling_step(num_micro_batches):
    recompute_profiling_times = 4
    min_profiling_steps = 5
    recompute_profiling_steps = recompute_profiling_times // num_micro_batches
    if recompute_profiling_times % num_micro_batches != 0:
        recompute_profiling_steps += 1
    return max(min_profiling_steps, recompute_profiling_steps)


def init_profiling_steps():
    num_micro_batches = get_num_microbatches()
    # cal profiling step
    recompute_profiling_steps = cal_profiling_step(num_micro_batches)
    # cal swap profiling step
    swap_profiling_steps = cal_swap_profiling_step(num_micro_batches)
    # init step
    AdaptiveStepMgr().init_steps(recompute_profiling_steps, swap_profiling_steps)
    print_rank_0(f"init profiling steps, recompute:{recompute_profiling_steps}, swap:{swap_profiling_steps}")


def update_swap_profiling_step_and_deep_list():
    # update swap profiling step
    swap_profiling_steps = cal_swap_profiling_step(get_num_microbatches())
    # update deep_list
    AdaptiveMemoryPrefetch().solve_prefetch_config()
    AdaptiveStepMgr().init_steps(AdaptiveStepMgr().recompute_profiling_steps, swap_profiling_steps)
    print_rank_0(f"update profiling steps, recompute:{AdaptiveStepMgr().recompute_profiling_steps}, swap:{swap_profiling_steps}, "
                 f"prefetch_deep_list:{AdaptiveMemoryPrefetch().prefetch_deep_list}, prefetch_hook_interval:{AdaptiveMemoryPrefetch().prefetch_hook_interval}")


def setup_adapt_memory_optimizer_wrapper(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(*args, **kwargs):
        models, optimizer, opt_param_scheduler = setup_model_and_optimizer(*args, **kwargs)

        optimizer.step = AdaptiveMemoryOpt().hook_adapt_mem_step(optimizer.step, models)
        AdaptiveMemoryProfiling().construct_and_register_profiling_hooks(models)

        init_profiling_steps()
        register_custom_hooks(models)

        AdaptiveMemoryPrefetch().solve_prefetch_config()
        # 绑核
        if "910B" in acl.get_soc_name() or "910A" in acl.get_soc_name():
            bind_cpus(torch.cuda.device_count(), torch.cuda.current_device(), 0)
        # 加载历史策略
        PolicyCacheManager().load_cache_file()

        return models, optimizer, opt_param_scheduler

    return wrapper
