# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import json
import itertools
import copy
import sys
import time
import math

import torch
import numpy as np

from mindspeed.core.auto_parallel.mm_search.help import (
    broadcast_communicate_list, 
    cal_throughput, 
    get_json, 
    INITIAL_CONFIG, 
    GPT_ARGS_PATH)
from mindspeed.core.auto_parallel.mm_search.pp_layer_search import pp_layer_search


def record_train_config(profile):
    for key in INITIAL_CONFIG:
        profile[key] = INITIAL_CONFIG[key]
    gpt_args = get_json(GPT_ARGS_PATH)
    for key in gpt_args:
        profile[key] = gpt_args[key]
    return profile


class AutoParallelSolver():
    def __init__(self, profile_data):
        if torch.cuda.is_available():
            self.max_available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        else:
            self.max_available_memory = 62000
        self.layer_name = ['vit_pre', 'vit', 'vit_post', 'llm_pre', 'llm', 'llm_post']
        print(f"[INFO] NPU available memory: {self.max_available_memory}")


    def update_profile(self, args, parallel_cfg, profile_data):
        update_profile_data = copy.deepcopy(profile_data)

        if args.use_distributed_optimizer:
            DP = parallel_cfg[2]
            
            for key in profile_data:
                if key in self.layer_name:
                    update_profile_data[key]['module_param'][1] = profile_data[key]['module_param'][1] / 12 * (4 + 8 / DP)
        
        return update_profile_data


    def cal_max_layer(self, profile_data):
        llm_available_memory = self.max_available_memory - sum(profile_data['llm_post']['module_param']) - profile_data['llm_post']['act_mem']
        last_stage_max_layer = math.floor(llm_available_memory / (sum(profile_data['llm']['module_param']))) + 1
        return last_stage_max_layer


    def trans_optimal_config(self, optimal_config, profile_data):
        parallel_config = optimal_config['parallel_config']
        optimal_config['parallel_config'] = {'PP': parallel_config[0], 
                                             'TP': parallel_config[1], 
                                             'DP': parallel_config[2], 
                                             'MBS': parallel_config[3]}
        
        layer_placement = optimal_config['layer_placement']
        sum_model_layer = profile_data['image_encoder.vision_encoder.num_layers'] + profile_data['text_decoder.num_layers']
        layer_placement.append(sum_model_layer)
        merge_layer_place = []
        merge_layer_place.append(int(layer_placement[0]))
        for i in range(1, len(layer_placement)):
            layer_num = int(layer_placement[i] - layer_placement[i - 1])
            merge_layer_place.append(layer_num)

        vit_layer_placement = [0] * optimal_config['parallel_config']['PP']
        llm_layer_placement = [0] * optimal_config['parallel_config']['PP']
        vit_layer_num = profile_data['image_encoder.vision_encoder.num_layers']
        llm_layer_num = profile_data['text_decoder.num_layers']
        for i, capacity in enumerate(merge_layer_place):
            a_count = min(vit_layer_num, capacity)
            vit_layer_placement[i] = a_count
            vit_layer_num -= a_count
            b_count = min(llm_layer_num, capacity - a_count)
            llm_layer_placement[i] = b_count
            llm_layer_num -= b_count
        optimal_config['layer_placement'] = {'vit_layer_placement': vit_layer_placement,
                                             'llm_layer_placement': llm_layer_placement}
        
        layer_recompute = optimal_config['layer_recompute']
        optimal_config['layer_recompute'] = {'vit_layer_recompute': layer_recompute[0],
                                             'llm_layer_recompute': layer_recompute[1]}
        return optimal_config


def solve_auto_parallel_mm(args, parallel_cfgs):
    if torch.distributed.get_rank() == 0:
        with open(f'model_profile.json', 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        solver = AutoParallelSolver(profile_data)

        optimal_config = {}
        optimal_throughput = 0
        for parallel_cfg in parallel_cfgs:
            print(f"[INFO] now to solve config {parallel_cfg}")
            
            cfg_profile_data = solver.update_profile(args, parallel_cfg, profile_data)
            
            last_stage_max_layer = solver.cal_max_layer(cfg_profile_data)
            print(f"[INFO] last stage max layer {last_stage_max_layer}")
            
            layer_placement, layer_recompute, e2e_time = pp_layer_search(parallel_cfg, cfg_profile_data, solver.max_available_memory, last_stage_max_layer)
            
            if e2e_time is None:
                continue

            per_npu_throughput = cal_throughput(e2e_time, cfg_profile_data, parallel_cfg)
            print(f"[INFO] per_npu throughput {per_npu_throughput}")
            
            if per_npu_throughput > optimal_throughput:
                optimal_config = {"parallel_config": parallel_cfg,
                                "layer_placement": layer_placement.tolist(), 
                                "layer_recompute": layer_recompute, 
                                "e2e_time": e2e_time, 
                                "throughput": per_npu_throughput}
                optimal_config = solver.trans_optimal_config(optimal_config, profile_data)
                optimal_throughput = per_npu_throughput
        
        print(f"optimal_config: {optimal_config}")
        return optimal_config
    return None

