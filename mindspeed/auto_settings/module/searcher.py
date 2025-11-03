import math
import os
import time

from mindspeed.auto_settings.module.memory_cost import MemoryCost
from collections import deque
from copy import deepcopy
from typing import Deque, Tuple, Optional, List
import traceback
import numpy as np
import pandas as pd

from mindspeed.auto_settings.config.search_config import SearchConfig, ExecutorFlag
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.memory_cost_black import MemoryCostBlack
from mindspeed.auto_settings.module.time_cost import TimeCost
from mindspeed.auto_settings.module.time_cost_black import TimeCostBlack
from mindspeed.auto_settings.profile.profiler import Profiler
from mindspeed.auto_settings.search_space import SearchSpace
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.utils import get_prof_dir, get_black_prof_file

"""
Searcher Strategy
"""


class Searcher(object):
    def __init__(self):
        pass

    def search(self, configs, topk):
        raise Exception("This method has not been implemented here.")


class BaseSearcher(Searcher):
    def __init__(self):
        super().__init__()
        self.search_spaces = SearchSpace()
        self.profiler = Profiler()
        self.memory_cost = MemoryCost()

    def get_logger(self):
        raise Exception("This method has not been implemented here.")

    def train_models(self, profile_results):
        self.memory_cost.train_models()

    def pre_search(self):
        """
        搜索前profiling建模
        """
        space_begin_time = time.time()
        pre_configs = self.search_spaces.build_pre_search_spaces()
        self.get_logger().info(">>>>>> Generate profiling config cost time: %s ms",
                         str((time.time() - space_begin_time) * 1000))
        profiling_and_parser_begin_time = time.time()
        profile_results = self.profiler.profile(pre_configs)
        self.get_logger().info(">>>>>> Profiling and parser cost time: %s ms",
                         str((time.time() - profiling_and_parser_begin_time) * 1000))
        self.train_models(profile_results)


class WhiteSearcher(BaseSearcher):
    def __init__(self):
        super().__init__()
        self.time_cost = TimeCost()
        self.logger = get_logger("WhiteSearcher")

    def get_logger(self):
        return self.logger

    def train_models(self, profile_results):
        super().train_models(profile_results)
        self.time_cost.train_models(profile_results)

    def get_memory_cost(self, config):
        """
        获取峰值预估内存
        """
        return self.memory_cost.get_memory_cost(config)

    def get_time_cost(self, config, memory_info):
        """
        获取预估耗时
        """
        return self.time_cost.get_time_cost(config, memory_info)

    def log_configs(self, configs):
        """
        对configs打日志
        """
        for cfg in configs:
            self.logger.info(
                f"Stage [1] pruned config: TP=[{cfg.tp}] PP=[{cfg.pp}] LAYERS_PER_VPP=[{cfg.layers_per_vpp}] "
                f"DP=[{cfg.dp}] CP=[{cfg.cp}] EP=[{cfg.ep}] ZeRO=[{cfg.zero1}]")

    def search(self, configs, topk):
        """
        线上搜索方法
        configs: 全量的搜索空间
        """
        white_begin_time = time.time()
        self.pre_search()

        system_config = get_system_config()
        device_mem_cap = system_config.memory_cap
        self.logger.info(f"Search: total_device_num: {system_config.search_world_size}")
        self.logger.info(f"Search: device_mem_cap: {device_mem_cap}")
        best_perf_cfg_map: Deque[Tuple[float, Optional[SearchConfig]]] = deque([(float("inf"), None)] * topk, topk)

        self.logger.info(f"Stage [1] pruned result: number of valid PTD configurations [{len(configs)}]")
        self.log_configs(configs)

        search_config_begin_time = time.time()
        for cfg in configs:
            self.logger.info("====================")
            self.logger.info(f"Looking at:\n\n{cfg}")
            memory_info = self.get_memory_cost(cfg)
            if memory_info["peak_memory"] > device_mem_cap:
                self.logger.info(f"OOM found, next!")
                continue
            try:
                time_info = self.get_time_cost(cfg, memory_info)
                new_perf = time_info["total_time"]
                recompute_layer = time_info["num_layers"]
                use_mc2 = time_info["use_mc2"]
                new_memory = memory_info["new_memory"]
                self.logger.debug(
                    f"after recompute, perf = {new_perf} and need_recompute = {memory_info['need_recompute']}")
                self.logger.debug(f"cur mem_estimated = {new_memory}, recompute_layer = {recompute_layer}")

                better_found = False
                for i, perf_cfg in enumerate(best_perf_cfg_map):
                    if new_perf < perf_cfg[0]:
                        better_found = True
                        cfg.performance = new_perf
                        cfg.memory = new_memory
                        cfg.recompute_num_layers = recompute_layer
                        cfg.use_ascend_mc2 = use_mc2 if cfg.tensor_model_parallel_size > 1 else False
                        self.logger.info(f"Search: SUCCESSFUL Better #{i} Config Found.")
                        self.logger.debug(f"Performance Estimation: {new_perf}.")
                        best_perf_cfg_map.pop()
                        best_perf_cfg_map.insert(i, (new_perf, deepcopy(cfg)))
                        break
                if not better_found:
                    self.logger.info(f"Sub-optimal performance, next!")
            except Exception as err:
                self.logger.warning(f"Search: ERROR during perf_modeling_calculation: {type(err).__name__}")
                traceback.print_exc()
        final_cfgs = [cfg for _, cfg in best_perf_cfg_map]
        self.logger.info(">>>>>> Search configuration cost time: %s ms",
                         str((time.time() - search_config_begin_time) * 1000))
        self.logger.info(">>>>>> Total execution cost time: %s ms",
                         str((time.time() - white_begin_time) * 1000))
        return final_cfgs


class BlackSearcher(BaseSearcher):

    def __init__(self):
        super().__init__()
        self.logger = get_logger("BlackSearcher")
        self.search_spaces = SearchSpace()
        self.profiler = Profiler()
        self.memory_black = MemoryCostBlack()
        self.time_black = TimeCostBlack()
        self.model_results = pd.DataFrame(columns=[
            'pp', 'tp', 'dp', 'ring_attention', 'ulysses', 'mbs', 'vpp', 'ep', 'peak_memory', 'e2e_time'
        ])

        self.output_path = os.path.join(get_system_config().work_dir, 'model_results.csv')

    def get_logger(self):
        return self.logger

    def search(self, configs: List[SearchConfig], topk):
        """
        黑盒搜索方案
        """
        self.pre_search()
        idx = 0
        while idx < len(configs):
            config = configs[idx]
            self.logger.info(f">>> current config: {config}")

            work_dir = os.path.join(get_system_config().work_dir, get_prof_dir(config))
            if os.path.exists(work_dir):
                os.makedirs(work_dir, exist_ok=True)
            micro_batch_size = config.micro_batch_size
            cropped_config = config.crop()
            cropped_config.micro_batch_size = 1
            cropped_config.prepare_for_profiling_black()
            cropped_config.prof_file = get_black_prof_file(cropped_config)
            # PP = 1, VPP = 1, MBS = 1
            self.logger.info(f"profiler cropped_with_pp_vpp_mbs config: {cropped_config}")
            self.profiler.run(get_prof_dir(cropped_config), config, ExecutorFlag.PROFILE_BLACK)

            peak_memory = self.memory_black.get_peak_memory(config)
            self.logger.info(f"the peak_mem of croped_mbs_config({config}) is {peak_memory}")
            if peak_memory > get_system_config().max_available_memory:
                tmp_search_space = deepcopy(configs)
                mem1 = self.memory_cost.get_memory_cost(cropped_config)
                mem1 = mem1["peak_memory"]
                for i in range(idx + 1, len(tmp_search_space)):
                    # 解决预估不准的问题
                    mem2 = self.memory_cost.get_memory_cost(tmp_search_space[i])
                    mem2 = mem2["peak_memory"]
                    if mem2 > mem1:
                        self.logger.info(f"==> remove config({tmp_search_space[i]}), mem1: {mem1} mem2: {mem2}")
                        configs.remove(tmp_search_space[i])
                idx += 1
                continue

            self.logger.info(f"profiler cropped config: {cropped_config}")
            cropped_config.micro_batch_size = micro_batch_size
            cropped_config.prepare_for_profiling_black()
            cropped_config.prof_file = get_black_prof_file(cropped_config)
            self.profiler.run(get_prof_dir(config), config, ExecutorFlag.PROFILE_BLACK)
            step_time = np.mean(self.time_black.get_iteration_time(config)) / 1e3  # ms
            peak_mem = self.memory_black.get_peak_memory(config)
            self.add_model_result(config, peak_mem, step_time)
            idx += 1

        self.model_results.to_csv(self.output_path, index=False)
        topk_config = self.get_top_k(topk=topk)
        self.model_results.to_csv(self.output_path, index=False)
        return topk_config

    def add_model_result(self, config: SearchConfig, peak_mem, cost_time):
        print(f'cost_time: {cost_time}')
        if math.isinf(cost_time.mean()):
            self.logger.warning(f"cost_time of Config-{config} is inf")
            return

        self.model_results.loc[len(self.model_results.index)] = [
            config.pipeline_model_parallel_size,
            config.tensor_model_parallel_size,
            config.data_parallel_size,
            config.ring_attention_size,
            config.ulysses_size,
            config.micro_batch_size,
            config.virtual_pipeline_model_parallel_size,
            config.expert_model_parallel_size,
            peak_mem,
            cost_time
        ]

    def get_top_k(self, topk=3, threshold=0.95):
        results: List[SearchConfig] = []
        available_memory = get_system_config().max_available_memory * threshold

        data_frame = self.model_results[self.model_results['peak_memory'] < available_memory]
        data_frame = data_frame.sort_values(by='e2e_time')
        data_frame = data_frame.reset_index(drop=True)
        topk = min(topk, len(data_frame.index))
        for i in range(topk):
            row = data_frame.loc[i]
            config = SearchConfig(
                pipeline_model_parallel_size=int(row['pp']),
                tensor_model_parallel_size=int(row['tp']),
                data_parallel_size=int(row['dp']),
                ring_attention_size=row['ring_attention'],
                ulysses_size=row['ulysses'],
                micro_batch_size=int(row['mbs']),
                virtual_pipeline_model_parallel_size=int(row['vpp']),
                expert_model_parallel_size=row['ep']
            )
            results.append(config)
        return results


class MixedSearcher(Searcher):

    def __init__(self):
        super().__init__()
        self.white = WhiteSearcher()
        self.black = BlackSearcher()
        self.white_topk = 30

    def set_white_topk(self, topk):
        self.white_topk = topk

    def search(self, configs, topk):
        """
        综合搜索策略
        """
        w_configs = self.white.search(configs, self.white_topk)
        b_configs = self.black.search(w_configs, topk)
        return b_configs
