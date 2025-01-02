# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
import copy
import time
import math
import random
from pathlib import Path
from typing import List, Literal
from abc import ABC, abstractmethod
import traceback

import numpy as np
import pandas as pd
import gpytorch

from .memory_model import MemoryCostModel
from .perf_model import TimeCostModel
from .operator_model_cache import model_manager
from .operator_model_cache import sample_cache
from .operator_model_cache import operator_cache
from .operator_model import ExactGPModel
from ..utils import logger
from ..utils.parallel_config import ParallelConfig
from ..utils.system_config import get_system_config
from ..utils.model_config import get_model_config
from ..utils.profiler_args import OperatorProfileArgs
from ..utils.profiler import Launch
from ..utils.data_type import TaskType
from ..utils.utils import get_cache_path


class Search(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model_results = pd.DataFrame(columns=[
            'pp', 'tp', 'dp', 'ring_attention', 'ulysses', 'mbs', 'vpp', 'ep', 'peak_memory', 'e2e_time'
        ])
        self.output_path = get_cache_path() + 'model_results.csv'

    def add_model_result(self, config: ParallelConfig, peak_mem, cost_time):
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
        results: List[ParallelConfig] = []
        available_memory = get_system_config().max_available_memory * threshold

        data_frame = self.model_results[self.model_results['peak_memory'] < available_memory]
        data_frame = data_frame.sort_values(by='e2e_time')
        data_frame = data_frame.reset_index(drop=True)
        topk = min(topk, len(data_frame.index))
        for i in range(topk):
            row = data_frame.loc[i]
            config = ParallelConfig(
                pipeline_model_parallel_size=int(row['pp']),
                tensor_model_parallel_size=int(row['tp']),
                data_parallel_size=int(row['dp']),
                ring_attention_size=int(row['ring_attention']),
                ulysses_size=int(row['ulysses']),
                micro_batch_size=int(row['mbs']),
                virtual_pipeline_model_parallel_size=int(row['vpp']),
                expert_model_parallel_size=int(row['ep'])
            )
            results.append(config)
        return results

    @staticmethod
    def print_search_space(label: str, search_space):
        logger.info(f"{'#' * 10} {label}")
        logger.info(f"count: {len(search_space)}")
        for config in search_space:
            logger.info(config)

    @abstractmethod
    def search(self, search_space):
        raise NotImplementedError('This method should be overridden by subclasses')


class SearchByFullPrecision(Search):
    def search(self, search_space: List[ParallelConfig]):
        idx = 0
        while idx < len(search_space):
            config = search_space[idx]
            cropped_config = config.crop_config()
            cropped_config.micro_batch_size = 1
            logger.info(f"start explore cropped_config: {cropped_config}")
            Launch.launch(OperatorProfileArgs(cropped_config), TaskType.OPERATOR_PROFILLING)

            tmp_config = copy.deepcopy(config)
            tmp_config.micro_batch_size = 1
            peak_memory = MemoryCostModel.get_peak_memory(tmp_config, 'black_box')
            logger.info(f"the peak_mem of croped_b_config({tmp_config}) is {peak_memory}")
            if peak_memory > get_system_config().max_available_memory:
                tmp_search_space = copy.deepcopy(search_space)
                mem1 = MemoryCostModel.get_peak_memory(tmp_config, 'white_box')
                for i in range(idx + 1, len(tmp_search_space)):
                    mem2 = MemoryCostModel.get_peak_memory(tmp_search_space[i], 'white_box')
                    if mem2 > mem1:
                        logger.info(f"==> remove config({tmp_search_space[i]}), mem1: {mem1} mem2: {mem2}")
                        search_space.remove(tmp_search_space[i])
                
                idx += 1
                continue

            cropped_config.micro_batch_size = config.micro_batch_size
            logger.info(f"start explore config: {config}")
            Launch.launch(OperatorProfileArgs(cropped_config), TaskType.OPERATOR_PROFILLING)
            step_time = np.mean(TimeCostModel.get_iteration_time(config, 'module_level')) / 1e3 # ms
            peak_mem = MemoryCostModel.get_peak_memory(config, 'black_box')
            self.add_model_result(config, peak_mem, step_time)
            idx += 1

        topk_config = self.get_top_k(topk=3)
        self.model_results.to_csv(self.output_path, index=False)
        return topk_config
    

class SearchByFastMode(Search):
    def __init__(self, stop_threshold=0.05):
        super().__init__()
        # 通过算子级模型建模的算子
        self.operators = [
            'MatMul', 
            'RmsNorm', 
            'RmsNormGrad', 
            'LayerNorm',
            'LayerNormGrad',
            'FlashAttentionScore', 
            'FlashAttentionScoreGrad'
        ]

        normalization = get_model_config().args.normalization
        if normalization == 'RMSNorm':
            self.operators.remove('LayerNorm')
            self.operators.remove('LayerNormGrad')
        else:
            self.operators.remove('RmsNorm')
            self.operators.remove('RmsNormGrad')

        self.stop_threshold = stop_threshold
        self.config_performances = {}
        self.exist_config = []
        self.e2e_log = pd.DataFrame()
        self.memory_threshold = 0.90
        self.topk = 3

    @staticmethod
    def find_csv(operator_profile, key='kernel_details'):
        csv_files = []
        for cf in list(Path(operator_profile).rglob('*.csv')):
            if key in str(cf):
                csv_files.append(os.path.abspath(str(cf)))
        if len(csv_files) <= 0:
            return None
        return sorted(csv_files)[0]

    def save(self, config: ParallelConfig, cost_time: float):
        self.e2e_log[str(config.to_list())] = cost_time

    def generate_config(self):
        def statistic_frequency(rest_config):
            freq_count = dict()
            for config in rest_config:
                if str(config) not in freq_count.keys():
                    freq_count[str(config)] = 0
                freq_count[str(config)] += 1
            
            for key, value in freq_count.items():
                logger.info("config: {} freq: {} explore_prob: {:.2f}".format(key, value, value / len(rest_config)))

        topk_indices = self.e2e_log.apply(lambda x: x.nsmallest(self.topk).index.tolist(), axis=1).values
        topk_indices_list = list()
        for indices in topk_indices:
            for config in indices:
                topk_indices_list.append(config)

        rest_config = [i for i in topk_indices_list if str(i) not in self.exist_config]
        logger.info(f"Rest best onfigs: {list(set(rest_config))}")
        statistic_frequency(rest_config)

        prop = len(rest_config) / len(topk_indices_list)
        if prop > self.stop_threshold:
            sample = random.choice(rest_config)
            self.exist_config.append(sample)
            return ParallelConfig.from_list(eval(sample))
        
        logger.info(f'Unexplored proportion: {prop} < stop_thd :{self.stop_threshold}, early stop triggered.')
        return None

    def train(self, train_profiling_file, train_operator_data):
        for operator in self.operators:
            logger.info(f"start fit {operator}")
            model = model_manager.get_cached_model(operator)
            if model is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    gpytorch.priors.NormalPrior(1e-3, 0.02)
                )
                model = ExactGPModel(operator=operator, likelihood=likelihood)
                model_manager.cache_model(model, operator)
            model.fit(train_profiling_file, train_operator_data)

    def load_base_model(self, model_dir):
        for operator in self.operators:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(gpytorch.priors.NormalPrior(1e-3, 0.02))
            model = ExactGPModel(operator=operator, likelihood=likelihood)
            try:
                model_manager.load_model(model, operator, model_dir)
            except Exception as e:
                logger.warning(f"{operator} load error")

    def rm_greater_mem_configs(self, search_spaces: List[ParallelConfig], config: ParallelConfig):
        mem1 = MemoryCostModel.get_peak_memory(config, 'white_box')
        for i in range(len(search_spaces), -1, -1, -1):
            mem2 = MemoryCostModel.get_peak_memory(search_spaces[i], 'white_box')
            if mem2 >= mem1:
                logger.info(f"remove config({search_spaces[i]}), mem1={mem1} mem2={mem2}")
                search_spaces.pop(i)
                self.e2e_log.remove(str(search_spaces[i].to_list()))

    def search(self, search_spaces: List[ParallelConfig]):
        start_time = time.time()
        explore_times = 0
        while ((time.time() - start_time) / 3600) < 8:
            # step1 对搜索空间中的并行配置进行算子级性能建模
            for config in search_spaces:
                cost_time = TimeCostModel.get_iteration_time(config, 'operator_level')
                self.save(config, cost_time)
                logger.info(f'complete model config({config}), cost_time: {np.mean(cost_time) / 1e3}')

            # step2 通过灰盒搜索算法推荐下一次探索的并行配置
            next_config = self.generate_config()
            logger.info(f"{explore_times + 1}-th next_config={next_config}")
            if next_config is None:
                break
            
            # step3 下一次探索
            explore_times += 1
            cropped_next_config = next_config.crop_config()
            Launch.launch(OperatorProfileArgs(cropped_next_config), TaskType.OPERATOR_PROFILLING)
            try:
                peak_mem = MemoryCostModel.get_peak_memory(next_config, method='black_box')
                if peak_mem > get_system_config().max_available_memory:
                    logger.info(f"remove next_config({next_config}) peak_mem={peak_mem}")
                    search_spaces.remove(next_config)
                    self.e2e_log.pop(str(next_config.to_list()))
                    self.rm_greater_mem_configs(search_spaces, next_config)
                    continue

                cost_time = np.mean(TimeCostModel.get_iteration_time(next_config, method='module_level')) / 1e3 # ms
                if not math.isinf(cost_time):
                    self.add_model_result(next_config, peak_mem, cost_time)
                
                logger.info(f'complete explore config({next_config}), cost_time: {cost_time}')

                # step4 更新算子级模型
                operator_profile_path = SearchByFastMode.find_csv(cropped_next_config.operator_profile_path)
                if not operator_profile_path:
                    logger.warning('not find kernel_details.csv')
                    sample_cache.clear_cache()
                    continue

                self.train(operator_profile_path, operator_cache.data_frame)
                sample_cache.clear_cache()
            except BaseException as e:
                logger.error(f"update operate model failed. exception info: {e}")

        topk_config = self.get_top_k(topk=3)
        self.model_results.to_csv(self.output_path, index=False)
        return topk_config