from typing import Deque, List, Optional, Tuple
from collections import deque
from copy import deepcopy
import pickle
import os
import sys
import traceback as tb



from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.search.stage_1_prune import stage_1_discrete_search_space_prune
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.utils.utils import get_prof_dir
from mindspeed.auto_tuning.utils.restricted_unpickler import restricted_loads
from mindspeed.auto_tuning.config.generate_profiling_configs import generate_profiling_configs


_logger = get_logger("search")


def search_demo(model_config: ModelConfig,
                perf_obj_function,
                working_dir: str,
                re_profiling_flag=True,
                recomp_cfg_list=None) -> [List[Optional[SearchConfig]], tuple]:
    device_mem_cap = Hardware().memory_limit
    _logger.info(f"Search: total_device_num: {Hardware().num_devices}")
    _logger.info(f"Search: device_mem_cap: {device_mem_cap}")
    best_perf_cfg_map: Deque[Tuple[float, Optional[SearchConfig]]] = deque([(float("inf"), None)] * 3, 3)

    stage_1_valid_ptd_configs = stage_1_discrete_search_space_prune(
        model_config,
        pod_limit=8
    )

    _logger.info(f"Stage [1] pruned result: number of valid PTD configurations [{len(stage_1_valid_ptd_configs)}]")
    for cfg in stage_1_valid_ptd_configs:
        _logger.info(f"Stage [1] pruned config: TP=[{cfg.tp}] PP=[{cfg.pp}] LAYERS_PER_VPP=[{cfg.layers_per_vpp}] DP=[{cfg.dp}] CP=[{cfg.cp}] EP=[{cfg.ep}] ZeRO=[{cfg.zero1}]")

    base_context = ""
    base_search_cfg = None
    for cfg in generate_profiling_configs(model_config):
        json_path = os.path.join(working_dir, f'{get_prof_dir(cfg)}.json')
        # find ep = 1 config
        if (not os.path.exists(json_path) or cfg.expert_model_parallel_size and
                cfg.expert_model_parallel_size != 1):
            continue
        try:
            with open(json_path, "rb") as file:
                base_context = restricted_loads(file)
                base_search_cfg = cfg
        except pickle.UnpicklingError as e:
            _logger.warning(f"Incorrect pickle format. UnpicklingError: {e}")
            raise e
        if base_context:
            break

    _logger.debug(f"success print base_context = {base_context}")
    uncovered_prof = []
    profile_count = [0]

    for cfg in stage_1_valid_ptd_configs:
        _logger.info("====================")
        _logger.info(f"Looking at:\n\n{cfg}")
        mem_estimated, _ = MemoryModeling.estimate(cfg)
        if mem_estimated <= device_mem_cap:
            try:
                perf, uncovered_prof, use_mc2 = perf_obj_function(cfg, working_dir, profile_count, re_profiling_flag)
            except Exception as err:
                _logger.warning(f"Search: ERROR during perf_modeling_calculation: {type(err).__name__}")
                tb.print_exc()

            context = ""
            json_path = os.path.join(working_dir, f'{get_prof_dir(cfg)}.json')
            if not os.path.exists(json_path):
                _logger.debug("success modeling context…………")
                context = get_context_by_ptd_config(base_context, base_search_cfg, cfg, model_config)
            else:
                try:
                    with open(json_path, "rb") as file:
                        context = restricted_loads(file)
                except pickle.UnpicklingError as e:
                    _logger.warning(f"Incorrect pickle format. UnpicklingError: {e}")
                    raise e
            _logger.debug(f"before recompute, perf = {perf} and memory = {mem_estimated}")
            _logger.debug(f"success enter recompute_solver and tp = {cfg.tensor_model_parallel_size} "
                          f"pp = {cfg.pipeline_model_parallel_size} "
                          f"layers_per_vpp={cfg.num_layers_per_virtual_pipeline_stage} "
                          f"dp = {cfg.data_parallel_size} cp = {cfg.context_parallel_size} "
                          f"ep = {cfg.expert_model_parallel_size} zero = {cfg.use_distributed_optimizer}")
            need_recompute, new_perf, add_mem, recompute_layer = full_recompute_solver(device_mem_cap - mem_estimated, context,
                                                                                       model_config, perf, cfg)
            new_memory = add_mem + mem_estimated
            _logger.debug(f"after recompute, perf = {new_perf} and need_recompute = {need_recompute}")
            _logger.debug(f"cur mem_estimated = {new_memory}, recompute_layer = {recompute_layer}")

            better_found = False
            for i, perf_cfg in enumerate(best_perf_cfg_map):
                if new_perf < perf_cfg[0]:
                    better_found = True
                    cfg.adaptive_recompute_device_swap = need_recompute
                    cfg.performance = new_perf
                    cfg.memory = new_memory
                    cfg.recompute_num_layers = recompute_layer
                    cfg.use_ascend_mc2 = use_mc2 if cfg.tensor_model_parallel_size > 1 else False
                    _logger.info(f"Search: SUCCESSFUL Better #{i} Config Found.")
                    _logger.debug(f"Performance Estimation: {new_perf}.")
                    best_perf_cfg_map.pop()
                    best_perf_cfg_map.insert(i, (new_perf, deepcopy(cfg)))
                    break
            if not better_found:
                _logger.info(f"Sub-optimal performance, next!")

        else:
            _logger.info(f"OOM found, next!")

    return [cfg for _, cfg in best_perf_cfg_map], uncovered_prof


def get_context_by_ptd_config(base_context, base_search_cfg, search_cfg, model_config):
    cur_cfg_seq_multi_mbs_div_tp_cp = (search_cfg.seq_length / search_cfg.tensor_model_parallel_size /
                                       search_cfg.context_parallel_size) * search_cfg.micro_batch_size
    base_cfg_seq_multi_mbs_div_tp_cp = (base_search_cfg.seq_length / base_search_cfg.tensor_model_parallel_size /
                                        base_search_cfg.context_parallel_size) * base_search_cfg.micro_batch_size
    cur_cfg_resize_time = cur_cfg_seq_multi_mbs_div_tp_cp / base_cfg_seq_multi_mbs_div_tp_cp
    context = deepcopy(base_context)

    cur_experts_num = 0 if model_config.num_experts is None \
        else model_config.num_experts // search_cfg.expert_model_parallel_size
    recursive_change_context(context, cur_cfg_resize_time, cur_experts_num)

    return context


def recursive_change_context(context, cur_cfg_resize_time, cur_experts_num):
    if "memory" in context:
        context['memory'] *= cur_cfg_resize_time
    if 'input' in context:
        context['input'] *= cur_cfg_resize_time
    if 'time' in context:
        context['time'] *= cur_cfg_resize_time

    check_prefix_name = 'prefix_name' in context and 'mlp' in context.get('prefix_name')
    check_layer = 'layers' in context and context['layers'][0]['name'] == '0'
    if check_prefix_name and check_layer:
        context['layers'] = context['layers'][:cur_experts_num]
    if "layers" not in context:
        return
    for layer_context in context["layers"]:
        recursive_change_context(layer_context, cur_cfg_resize_time, cur_experts_num)


class ToyModel(object):
    def __init__(self):
        return


def perf_test_obj_function(search_config):
    return


def mem_test_toy_function(search_config):
    return


def get_first_layer_context(context):
    if "memory" in context:
        return context

    if "layers" not in context:
        return None
    for layer_context in context["layers"]:
        first_layer_context = get_first_layer_context(layer_context)
        if first_layer_context is not None:
            return first_layer_context
    return None


def memory_time_rate(ele):
    if ele["memory"] - ele["input"] == 0:
        return sys.maxsize
    return ele["time"] / (ele["memory"] - ele["input"])


def full_recompute_solver(oom_cap, model_context, model_cfg, perf, search_config):
    if search_config.layers_per_vpp:
        num_model_chunks = search_config.num_layers // search_config.layers_per_vpp // search_config.pp
        layers_per_vpp = search_config.layers_per_vpp
    else:
        num_model_chunks = 1
        layers_per_vpp = model_cfg.num_layers // search_config.pp
    warmup_micro_batchs, total_num_micro_batches = get_num_warmup_micro_batches(num_model_chunks, search_config,
                                                                                model_cfg)
    ret_list = []
    find_recompute_layer(model_context, ret_list)
    layer_module = ret_list[0]

    release_mem = 0
    time_cost = 0
    num_layers = model_cfg.num_layers // search_config.pp
    ret_list.sort(key=memory_time_rate, reverse=True)
    need_recompute = True
    memory_per_layer = layer_module["memory"] - layer_module["input"]
    # 1.No full recompute
    max_release_mem = warmup_micro_batchs * layers_per_vpp * memory_per_layer - memory_per_layer

    if max_release_mem <= oom_cap:
        return False, perf - total_num_micro_batches * num_layers * layer_module["time"], max_release_mem, 0

    if search_config.layers_per_vpp:
        # 2.Situation under per pp stage and per mbs recompute layers <= layers_per_vpp
        max_release_mem = (num_model_chunks - 1) * search_config.pp * layers_per_vpp * memory_per_layer
        if max_release_mem <= oom_cap:
            layer_calculate = (oom_cap - max_release_mem) // ((2 * search_config.pp - 1) * memory_per_layer)
            release_mem += (2 * search_config.pp - 1) * layer_calculate * memory_per_layer + max_release_mem - memory_per_layer
            time_cost += (num_layers - layers_per_vpp + layer_calculate) * total_num_micro_batches * layer_module["time"]
            return True, perf - time_cost, release_mem, layers_per_vpp - layer_calculate

        # Only consider layers temporarily
        layer_calculate = (oom_cap // (memory_per_layer * search_config.pp))
        release_mem += layer_calculate * memory_per_layer * search_config.pp
        if layer_calculate < num_layers:
            release_mem -= memory_per_layer
        time_cost += total_num_micro_batches * layer_calculate * layer_module["time"]
        return need_recompute, perf - time_cost, release_mem, num_layers - layer_calculate

    else:
        layer_calculate = (oom_cap // (memory_per_layer * search_config.pp))
        release_mem += layer_calculate * memory_per_layer * search_config.pp
        if layer_calculate < num_layers:
            release_mem -= memory_per_layer
        time_cost += total_num_micro_batches * layer_calculate * layer_module["time"]
        return need_recompute, perf - time_cost, release_mem, num_layers - layer_calculate


def get_num_warmup_micro_batches(num_model_chunks, search_config, model_cfg):
    pipeline_parallel_size = search_config.pp
    data_parallel_size = search_config.dp
    num_microbatches = model_cfg.gbs // (search_config.mbs * data_parallel_size)

    if pipeline_parallel_size <= 1:
        return 1, num_microbatches

    pipeline_parallel_size = pipeline_parallel_size
    pipeline_parallel_rank = 0
    total_num_micro_batches = num_microbatches * num_model_chunks
    if num_model_chunks == 1:
        num_warmup_micro_batches = pipeline_parallel_size - pipeline_parallel_rank - 1

    else:
        num_warmup_micro_batches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_micro_batches += (num_model_chunks - 1) * pipeline_parallel_size
    num_warmup_micro_batches += 1
    num_warmup_micro_batches = min(num_warmup_micro_batches, total_num_micro_batches)
    return num_warmup_micro_batches, num_microbatches


def find_recompute_layer(context, ret_list):
    if "memory" in context and context["input"] <= context["memory"]:
        layer_dict = {"memory": context["memory"], "time": context["time"],
                      "input": context["input"], "prefix_name": context["prefix_name"], "name": context["name"]}
        ret_list.append(layer_dict)

        # layer module the first to be appened
    if "layers" not in context:
        return
    for layer_context in context["layers"]:
        find_recompute_layer(layer_context, ret_list)
