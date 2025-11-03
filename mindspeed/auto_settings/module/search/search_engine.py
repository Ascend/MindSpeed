from typing import Deque, List, Optional, Tuple
from collections import deque
from copy import deepcopy
import os
import sys
import traceback as tb

from mindspeed.auto_settings.utils.logger import get_logger, change_stream_handler
from mindspeed.auto_settings.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.module.search.stage_1_prune import stage_1_discrete_search_space_prune
from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.utils.utils import get_prof_dir
from mindspeed.auto_settings.utils.file_utils import restricted_read
from functools import partial
from multiprocessing import Pool, JoinableQueue, Process, Event, queues, Manager
from io import StringIO
import time
from mindspeed.auto_settings.module.model_performance import ModelPerformance
from mindspeed.auto_settings.auto_settings import SingleModel


class SpaceSearch:
    def __init__(self):
        self._logger = get_logger("search")
        self.perf_cfg_map: Deque[Tuple[float, Optional[SearchConfig]]] = None
    
    def _space_search(self, models: list[SingleModel], cpu_num: int):
        logger = get_logger("search_logger")
        #return 0
        model = models[0]

        world_size = models[0].model_settings.search_world_size
        search_cfg_start_time = time.time()
        best_cfgs = self.search_demo(model=model)
        search_cfg_end_time = time.time()


        logger.info(">>>>>> Search_cfg cost time: %s ms",
                    str((search_cfg_end_time - search_cfg_start_time) * 1000))

        logger.info("<==========Final config generated==========>")
        logger.info("The recommended configs are:")
        for i, final_cfg in enumerate(best_cfgs):
            if final_cfg:
                logger.info("<==========Top #%s config==========>", str(i))
                logger.info("\n %s", str(final_cfg))
            logger.info("<==========Launch training==========>")

    def search_demo(
        self, 
        model: SingleModel,
        re_profiling_flag=True
    ) -> [List[Optional[SearchConfig]], tuple]:
        mem_model = model.memory_model
        perfmodel = model.model_performance
        setting = model.model_settings
        model_config = model.model_config
        working_dir = model.model_settings.work_dir,
        device_mem_cap = get_system_config().memory_cap
        self._logger.info(f"Search: total_device_num: {get_system_config().search_world_size}")
        self._logger.info(f"Search: device_mem_cap: {device_mem_cap}")
        best_perf_cfg_map: Deque[Tuple[float, Optional[SearchConfig]]] = deque([(float("inf"), None)] * 3, 3)

        stage_1_valid_ptd_configs = stage_1_discrete_search_space_prune(model_config)

        self._logger.info(f"Stage [1] pruned result: number of valid PTD configurations [{len(stage_1_valid_ptd_configs)}]")
        for cfg in stage_1_valid_ptd_configs:
            self._logger.info(f"Stage [1] pruned config: TP=[{cfg.tp}] PP=[{cfg.pp}] LAYERS_PER_VPP=[{cfg.layers_per_vpp}] DP=[{cfg.dp}] CP=[{cfg.cp}] EP=[{cfg.ep}] ZeRO=[{cfg.zero1}]")

        uncovered_prof = []
        profile_count = [0]
        fw_performance = 0

        for cfg in stage_1_valid_ptd_configs:
            self._logger.info("====================")
            self._logger.info(f"Looking at:\n\n{cfg}")
            recompute_mem, peak_stage_mem, optimizer_peak = mem_model.estimate(cfg)
            if max(peak_stage_mem, optimizer_peak) <= device_mem_cap:
                try:
                    perf, uncovered_prof, use_mc2, fw_performance = model.model_performance.performance(
                        cfg, working_dir, profile_count, re_profiling_flag
                    )
                except Exception as err:
                    self._logger.warning(f"Search: ERROR during perf_modeling_calculation: {type(err).__name__}")
                    tb.print_exc()

                self._logger.debug(f"before recompute, perf = {perf} and memory = {peak_stage_mem}")
                self._logger.debug(f"success enter recompute_solver and tp = {cfg.tensor_model_parallel_size} "
                            f"pp = {cfg.pipeline_model_parallel_size} "
                            f"layers_per_vpp={cfg.num_layers_per_virtual_pipeline_stage} "
                            f"dp = {cfg.data_parallel_size} cp = {cfg.context_parallel_size} "
                            f"ep = {cfg.expert_model_parallel_size} zero = {cfg.use_distributed_optimizer}")
                need_recompute, new_perf, add_mem, recompute_layer = self.full_recompute_solver(
                    device_mem_cap - peak_stage_mem, model_config, perf, cfg, recompute_mem, fw_performance
                )
                new_memory = add_mem + peak_stage_mem
                self._logger.debug(f"after recompute, perf = {new_perf} and need_recompute = {need_recompute}")
                self._logger.debug(f"cur mem_estimated = {new_memory}, recompute_layer = {recompute_layer}")

                better_found = False
                for i, perf_cfg in enumerate(best_perf_cfg_map):
                    if new_perf < perf_cfg[0]:
                        better_found = True
                        cfg.performance = new_perf
                        cfg.memory = new_memory
                        cfg.recompute_num_layers = recompute_layer
                        cfg.use_ascend_mc2 = use_mc2 if cfg.tensor_model_parallel_size > 1 else False
                        self._logger.info(f"Search: SUCCESSFUL Better #{i} Config Found.")
                        self._logger.debug(f"Performance Estimation: {new_perf}.")
                        best_perf_cfg_map.pop()
                        best_perf_cfg_map.insert(i, (new_perf, deepcopy(cfg)))
                        break
                if not better_found:
                    self._logger.info(f"Sub-optimal performance, next!")

            else:
                self._logger.info(f"OOM found, next!")

        return [cfg for _, cfg in best_perf_cfg_map]
    
    
    def full_recompute_solver(self, oom_cap, model_cfg: ModelConfig, perf, search_config, fw_memory, fw_performance):
        if search_config.layers_per_vpp:
            num_model_chunks = search_config.num_layers // search_config.layers_per_vpp // search_config.pp
            layers_per_vpp = search_config.layers_per_vpp
        else:
            num_model_chunks = 1
            layers_per_vpp = model_cfg.num_layers // search_config.pp
        warmup_micro_batchs, total_num_micro_batches = self.get_num_warmup_micro_batches(num_model_chunks, search_config,
                                                                                    model_cfg)
        release_mem = 0
        time_cost = 0
        num_layers = model_cfg.num_layers // search_config.pp
        need_recompute = True
        memory_per_layer = fw_memory
        max_release_mem = warmup_micro_batchs * layers_per_vpp * memory_per_layer - memory_per_layer
    
        if max_release_mem <= oom_cap:
            return False, perf - total_num_micro_batches * num_layers * fw_performance, max_release_mem, 0
    
        if search_config.layers_per_vpp:
            max_release_mem = (num_model_chunks - 1) * search_config.pp * layers_per_vpp * memory_per_layer
            if max_release_mem <= oom_cap:
                layer_calculate = (oom_cap - max_release_mem) // ((2 * search_config.pp - 1) * memory_per_layer)
                release_mem += (2 * search_config.pp - 1) * layer_calculate * memory_per_layer + max_release_mem - memory_per_layer
                time_cost += (num_layers - layers_per_vpp + layer_calculate) * total_num_micro_batches * fw_performance
                return True, perf - time_cost, release_mem, layers_per_vpp - layer_calculate
    
            layer_calculate = (oom_cap // (memory_per_layer * search_config.pp))
            release_mem += layer_calculate * memory_per_layer * search_config.pp
            if layer_calculate < num_layers:
                release_mem -= memory_per_layer
            time_cost += total_num_micro_batches * layer_calculate * fw_performance
            return need_recompute, perf - time_cost, release_mem, num_layers - layer_calculate
    
        else:
            layer_calculate = (oom_cap // (memory_per_layer * search_config.pp))
            release_mem += layer_calculate * memory_per_layer * search_config.pp
            if layer_calculate < num_layers:
                release_mem -= memory_per_layer
            time_cost += total_num_micro_batches * layer_calculate * fw_performance
            return need_recompute, perf - time_cost, release_mem, num_layers - layer_calculate
    
    
    def get_num_warmup_micro_batches(self, num_model_chunks, search_config, model_cfg):
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
    