# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import math
import types
from functools import wraps
import torch
import torch_npu

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import parallel_state

from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import AdaptiveRecompute
from mindspeed.core.memory.swap_attention.swap_attention_apply import AdaptiveRecomputeSwapReg


def setup_model_and_optimizer_wrapper(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(*args, **kargs):
        models, optimizer, opt_param_scheduler = setup_model_and_optimizer(*args, **kargs)
        profile_step = get_adaptive_recompute_profiling_step()

        recomputing = AdaptiveRecomputeSwap()
        recomputing.set_profiling_step(profile_step)
        recomputing.get_num_warmup_micro_batches(len(models))

        if isinstance(models, list):
            for index, model in enumerate(models):
                recomputing.construct_context_recursive("module" + str(index), model, recomputing.context, True)
        else:
            recomputing.construct_context_recursive("module", models, recomputing.context, True)

        recomputing.prefetch_hook(models)
        print_rank_0("ADAPTIVE-RECOMPUTE: successfully hooking module")
        return models, optimizer, opt_param_scheduler

    return wrapper


def allowed_recomputing_swap_module_wrapper(allowed_recomputing_module):
    recomputing = AdaptiveRecomputeSwap()
    recomputing.add_allowed_recomputing_module(allowed_recomputing_module)


def get_adaptive_recompute_profiling_step():
    all_args = get_args()
    max_profiling_step = all_args.train_iters // 10
    profiling_step = getattr(all_args, 'adaptive-recompute-profiling-step', 10)
    adaptive_recompute_device_size = getattr(all_args, 'adaptive-recompute-device-size', -1)
    adaptive_recompute_device_swap = getattr(all_args, 'adaptive-recompute-device-swap', False)
    if profiling_step < 5 or profiling_step > max_profiling_step:
        print_rank_0(f"[WARNING] consider set \"adaptive-recompute-profiling-step\" value >=5"
                     f"and <={max_profiling_step}, or remove it.")
    if profiling_step <= 0:
        print_rank_0("[WARNING] \"adaptive-recompute-profiling-step\" value can not <=0, will use default value 10.")
        profiling_step = 10
    print_rank_0(
        "success to activate adaptive recompute train: adaptive-recompute-device-swap={}, adaptive-recompute-device-size={}, "
        "adaptive-recompute-profiling-step={}".format(adaptive_recompute_device_swap,
                                                      adaptive_recompute_device_size, profiling_step))
    return profiling_step


class AdaptiveRecomputeSwap(AdaptiveRecompute):
    """Memory optimization handler combining activation recomputation and tensor prefetching."""
    adaptive_recomputing = None

    def __init__(self):
        super().__init__()
        self.interval = 0
        self.threshold_prefetch = 0
        self.num_prefetch = 0
        self.num_layers = 0

    def prefetch_hook(self, models):
        """
        Main entry point for applying prefetch/recompute policies to model layers.
        Args:
            models (nn.Module): Target model to apply memory optimizations
        """
        all_args = get_args()
        all_args.adaptive_recompute_device_size = getattr(all_args, 'adaptive-recompute-device-size', -1)
        self.reset_modules()
        swap_modules = all_args.swap_modules
        vpp = all_args.virtual_pipeline_model_parallel_size if all_args.virtual_pipeline_model_parallel_size else 1
        print_rank_0("ADAPTIVE-PREFETCH: Start applying policy to the model")
        config = {
            "pre_layer_full_name": "",
            "pre_layer_ctx": {},
            "cur_layer_name": "module",
            "swap_modules": swap_modules,
        }
        prefetch_recompute_group, interval, num_prefetch, swap_noop_layers = self.solve_prefetch_policy()
        print(f"[DEBUG] swap_list： {prefetch_recompute_group[0]},"
              f" prefetch_list： {prefetch_recompute_group[1]},"
              f" recompute_list： {prefetch_recompute_group[2]}")
        for i in prefetch_recompute_group[0]:
            if not any(filter(None, i)):
                vpp -= 1
        prefetch_args = [prefetch_recompute_group[0], vpp, interval, num_prefetch]
        AdaptiveRecomputeSwapReg().register_recursive_apply_prefetch(config, models, self.context, prefetch_recompute_group, prefetch_args)

    def solve_prefetch_policy(self):
        all_args = get_args()
        noop_layers = getattr(all_args, 'noop_layers', None)
        noop_layers = list(noop_layers) if isinstance(noop_layers, set) else []
        cur_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        cur_pp_noop_layers = self.get_cur_stage_noop_layers(noop_layers, cur_pp_rank)
        recompute_num_layers = all_args.recompute_num_layers or 0
        pp_size = all_args.pipeline_model_parallel_size or 1
        vpp_size = all_args.virtual_pipeline_model_parallel_size or 1
        per_pp_layers = all_args.num_layers // pp_size
        per_vpp_layers = all_args.num_layers_per_virtual_pipeline_stage or per_pp_layers
        if not getattr(all_args, 'enable_recompute_layers_per_pp_rank', False):
            if recompute_num_layers >= per_vpp_layers:
                recompute_num_layers = per_pp_layers
            else:
                recompute_num_layers *= vpp_size
        else:
            if recompute_num_layers >= per_pp_layers:
                recompute_num_layers = per_pp_layers
        if all_args.recompute_method == 'block':
            self.num_prefetch = recompute_num_layers
        elif all_args.recompute_method == 'uniform':
            recompute_num_layers = per_pp_layers
            self.num_prefetch = recompute_num_layers
        else:
            self.num_prefetch = per_pp_layers
        self.interval = 0
        if vpp_size > 1:
            return self.granular_module_allocation(vpp_size, recompute_num_layers, cur_pp_noop_layers)
        else:
            swap_list, recompute_list = [], []
            for i in range(self.num_prefetch):
                if i + cur_pp_rank * per_pp_layers not in cur_pp_noop_layers:
                    swap_list.append(str(i))
                else:
                    swap_list.append('')
            for i in range(recompute_num_layers):
                if i + cur_pp_rank * per_pp_layers not in cur_pp_noop_layers:
                    recompute_list.append(str(i))
                else:
                    recompute_list.append('')

            prefetch_list = swap_list
            prefetch_recompute_group = [[swap_list], [prefetch_list], [recompute_list]]
            return [prefetch_recompute_group, 0, len(prefetch_list), cur_pp_noop_layers]

    def get_cur_stage_noop_layers(self, noop_layers, cur_pp_rank):
        all_args = get_args()
        cur_pp_noop_layers = []
        pp_size = all_args.pipeline_model_parallel_size or 1
        layers_per_pp = all_args.num_layers // pp_size
        vpp_layer = all_args.num_layers_per_virtual_pipeline_stage or layers_per_pp
        vpp_layers = vpp_layer * pp_size
        for i in noop_layers:
            pp_id = (i % vpp_layers) // vpp_layer
            if pp_id == cur_pp_rank:
                cur_pp_noop_layers.append(i)
        return cur_pp_noop_layers

    def granular_module_allocation(self, vpp_size, recompute_num_layers, cur_pp_noop_layers):
        swap_list = []
        recompute_list = []
        args = get_args()
        cur_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = args.pipeline_model_parallel_size or 1
        vpp_layer = args.num_layers_per_virtual_pipeline_stage
        if self.num_prefetch <= vpp_size:
            swap_list = [['0'] if i < self.num_prefetch else [''] for i in range(vpp_size)]
        else:
            for chunk in range(vpp_size):
                chunk_swap_layer = ['0']
                for layer_id in range(vpp_size, self.num_prefetch):
                    if layer_id % vpp_size == chunk:
                        chunk_swap_layer.append(f'{layer_id // vpp_size}')
                swap_list.append(chunk_swap_layer)

        if recompute_num_layers <= vpp_size:
            recompute_list = [['0'] if i < recompute_num_layers else [''] for i in range(vpp_size)]
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and getattr(args, 'reduce_recompute_for_last_chunk', False):
                recompute_list[-1] = ['']
        else:
            for chunk in range(vpp_size):
                chunk_recompute_layer = ['0']
                for layer_id in range(vpp_size, recompute_num_layers):
                    if layer_id % vpp_size == chunk:
                        chunk_recompute_layer.append(f'{layer_id // vpp_size}')
                recompute_list.append(chunk_recompute_layer)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and getattr(args, 'reduce_recompute_for_last_chunk', False):
                if recompute_list[-1][-1] == str(args.num_layers_per_virtual_pipeline_stage - 1):
                    recompute_list[-1].pop()
                    if len(recompute_list[-1]) == 0:
                        recompute_list[-1].append('')
        for vpp in range(vpp_size):
            vpp_layers = swap_list[vpp]
            for i in range(len(vpp_layers)):
                layer_id = vpp * vpp_layer * pp_size + i + vpp_layer * cur_pp_rank
                if layer_id in cur_pp_noop_layers:
                    swap_list[vpp][i] = ''
                    if len(recompute_list[vpp]) >= i + 1:
                        recompute_list[vpp][i] = ''

        prefetch_list = swap_list
        interval = 0
        prefetch_recompute_group = [swap_list, prefetch_list, recompute_list]
        return [prefetch_recompute_group, interval, self.num_prefetch, cur_pp_noop_layers]