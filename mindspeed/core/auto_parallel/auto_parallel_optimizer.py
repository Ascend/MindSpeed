# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import math
import random
import ast
from pathlib import Path

import pandas as pd
import gpytorch
from megatron.training.global_vars import get_args

from mindspeed.core.auto_parallel import (
    model_manager,
    sample_cache,
    operator_cache,
)
from mindspeed.core.auto_parallel.auto_parallel_rectify import ExactGPModel
from mindspeed.core.auto_parallel.auto_parallel_model import TimeCostModel
from mindspeed.core.auto_parallel.auto_parallel_profiling import (
    BaseLaunch,
    DistributedOperateProfiler,
    DistributedPerformanceProfiler
)


class SearchByGreyBox:
    def __init__(self, stop_threshold=0.05):
        self.operators = [
            'MatMul', 
            'RmsNorm', 
            'RmsNormGrad', 
            'LayerNorm',
            'LayerNormGrad',
            'FlashAttentionScore', 
            'FlashAttentionScoreGrad'
        ]

        args = get_args()
        if args.normalization == 'RMSNorm':
            self.operators.remove('LayerNorm')
            self.operators.remove('LayerNormGrad')
        else:
            self.operators.remove('RmsNorm')
            self.operators.remove('RmsNormGrad')

        self.stop_threshold = stop_threshold
        self.config_performances = {}
        self.exist_config = []
        self.e2e_log = pd.DataFrame()

    @staticmethod
    def find_csv(operator_profile, key='kernel_details'):
        csv_files = []
        for cf in list(Path(operator_profile).rglob('*.csv')):
            if key in str(cf):
                csv_files.append(os.path.abspath(str(cf)))
        if len(csv_files) <= 0:
            print(f"not find kernel_details.csv")
            return None
        return sorted(csv_files)[0]

    @staticmethod
    def theory_modeling(config):
        base_launch = BaseLaunch()
        base_launch.update_args(config)
        cost_time = TimeCostModel().get_iteration_time()
        base_launch.recover_args()
        return cost_time

    def save(self, config, cost_time):
        self.e2e_log[str(config)] = cost_time

    def generate_config(self):
        best_config = self.e2e_log.apply(lambda col: col.idxmin(), axis=1).values
        rest_config = [i for i in best_config if str(i) not in self.exist_config]
        prop = len(rest_config) / len(best_config)
        if prop > self.stop_threshold:
            sample = random.choice(rest_config)
            self.exist_config.append(sample)
            return ast.literal_eval(sample)
        print(f'Unexplored proportion: {prop} < stop_thd :{self.stop_threshold}, early stop triggered.')
        return None

    def train(self, train_profiling_file, train_operator_data):
        for operator in self.operators:
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
            except Exception:
                print(f"{operator} load error")

    def search(self, args, search_spaces):
        start_time = time.time()
        self.load_base_model(os.path.dirname(os.path.abspath(__file__)) + os.sep + 'noise_predict_ckpt')
        while ((time.time() - start_time) / 3600) < 8 \
            and len(self.config_performances) < len(search_spaces):
            for config in search_spaces:
                cost_time = SearchByGreyBox.theory_modeling(config)
                self.save(config, cost_time)
                print(f"complete model config: {config}", flush=True)

            next_config = self.generate_config()
            if next_config is None:
                break
            print(f"next_config={next_config}", flush=True)

            operator_profile_path, analyse_thread = DistributedOperateProfiler().launch(next_config)
            duration_time = DistributedPerformanceProfiler().launch(next_config)
            self.config_performances[duration_time] = str(next_config)
            if math.isinf(duration_time):
                search_spaces.remove(next_config)
            if analyse_thread is not None:
                analyse_thread.join()
            
            operator_data = operator_cache.data_frame
            operator_profile = SearchByGreyBox.find_csv(operator_profile_path)
            if operator_profile is not None:
                print(f"operator_data: {operator_data}\noperator_profile: {operator_profile}")
                self.train(operator_profile, operator_data)
            sample_cache.clear_cache()
        
        model_manager.save_models('final_model')
        min_key = min(self.config_performances.keys())
        return ast.literal_eval(self.config_performances.get(min_key)), min_key
