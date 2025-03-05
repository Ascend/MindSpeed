# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import time
import copy
import sys
import json

import torch

from megatron.training import get_args
from mindspeed.core.auto_parallel import set_kv_store
from mindspeed.core.auto_parallel.mm_search.help import get_json, save_json, GPT_ARGS_PATH
from mindspeed.core.auto_parallel.mm_search.profiling import DistributedPerformanceProfiler
from mindspeed.core.auto_parallel.mm_search.solver import solve_auto_parallel_mm
from mindspeed.core.auto_parallel.mm_search.memory_modeling import get_model_total_static_memory, parallel_cluster_is_oom


class SearchByProfile:
    def __init__(self):
        self.merge_config_list = {}


    def get_gpt_args(self, args):
        gpt_args = {}
        world_size = args.nproc_per_node * args.nnodes
        tp = getattr(args, "tensor_model_parallel_size", 1)
        pp = getattr(args, "pipeline_model_parallel_size", 1)
        cp = getattr(args, "context_parallel_size", 1)
        dp = world_size / tp / pp / cp
        grad_acc_step = int(args.global_batch_size / args.micro_batch_size / dp)
        gpt_args['grad_acc_step'] = grad_acc_step
        save_json(GPT_ARGS_PATH, gpt_args)


    def merge_config(self, args, search_spaces):
        search_spaces_backup = copy.deepcopy(search_spaces)
        world_size = args.nproc_per_node * args.nnodes
        configs = []
        for ind, cfg in enumerate(search_spaces_backup):
            cfg[0] = 4                                      # pp
            cfg[2] = world_size // (cfg[0] * cfg[1])        # dp 
            if cfg[2] < 1:
                continue
            if cfg not in configs:
                configs.append(cfg)
                self.merge_config_list[tuple(cfg)] = [search_spaces[ind], ]
            else:
                self.merge_config_list[tuple(cfg)].append(search_spaces[ind])
        print("[INFO] merge config list", self.merge_config_list)

        return configs


    def search(self, args, search_spaces):
        self.get_gpt_args(args)
        merge_cfg = self.merge_config(args, search_spaces)

        opt_config = []
        run_throughput = 0
        for config in merge_cfg:
            print(f"[INFO] now profile config: {config}")

            status_code = 0
            status_code += DistributedPerformanceProfiler().launch(config, 'profiling_stage_1')
            status_code += DistributedPerformanceProfiler().launch(config, 'profiling_stage_2')

            if status_code == 0:
                parallel_split_config = self.merge_config_list[tuple(config)]
                print(f"[INFO] now solve cfg: {parallel_split_config}")
                
                optimal_config = solve_auto_parallel_mm(args, parallel_split_config)
                if optimal_config and optimal_config['throughput'] > run_throughput:
                    run_throughput = optimal_config['throughput']
                    opt_config = optimal_config

        opt_config_json = json.dumps(opt_config)
        with open(f'auto_parallel_search_optimal_config.json', 'w') as f:
            f.write(opt_config_json)
        print(f"[INFO] finally opt config: {opt_config}")


    @staticmethod
    def build_initial_spaces(args):
        world_size = args.simulated_nproc_per_node * args.simulated_nnodes
        device_count = args.simulated_nproc_per_node

        solutions = []
        for pp in range(1, world_size + 1):
            if world_size % pp != 0:
                continue

            for i in range(device_count):
                tp = 2 ** i
                if tp > device_count or tp > (world_size // pp):
                    break
                if (args.num_query_groups > 1 and args.num_query_groups % tp != 0) \
                    or (args.num_attention_heads % tp != 0):
                    break

                dp = world_size // (pp * tp)
                dp_group_batch_size = args.global_batch_size // dp
                for num_mb in range(1, dp_group_batch_size + 1):
                    if dp_group_batch_size % num_mb != 0:
                        continue

                    mbs = dp_group_batch_size // num_mb
                    if mbs > 2:
                        continue

                    solutions.append([pp, tp, dp, mbs])

        return solutions


    @staticmethod
    def filter_invalid_configs(args, search_spaces):
        rough_filter_configs = []
        for config in search_spaces:
            static_mem = get_model_total_static_memory(args, config)
            print(f"config: {config} static_mem: {static_mem}", flush=True)
            # PP将多模态网络分成4个部分
            if not parallel_cluster_is_oom(args, config, static_mem) and config[0] <= 16 and config[1] <= args.nproc_per_node / 4:
                rough_filter_configs.append(config)
        print(f"[INFO] finish static memory filter config {rough_filter_configs}")

        return rough_filter_configs


def monitor_train_task():
    while True:
        print(f"monitor next task...", flush=True)
        message = torch.tensor([0 for _ in range(5)], dtype=torch.int)
        torch.distributed.broadcast(message, src=0)
        task_type = message[-1].item()
        config = [m.item() for m in message[:-1]]
        if task_type == -1:
            break
        elif task_type == 0:
            DistributedPerformanceProfiler().launch(config)


def auto_parallel_mm_search_optimal_config(args):
    set_kv_store(args)
    # set cluster communication
    init_method = 'tcp://{}:{}'.format(args.master_addr, int(args.master_port) + 1)
    torch.distributed.init_process_group(
        backend=torch.distributed.Backend.GLOO,
        init_method=init_method,
        rank=args.node_rank,
        world_size=args.nnodes
    )

    if args.node_rank == 0:
        search_space = SearchByProfile().build_initial_spaces(args)
        print(f"[INFO] len(init_search_space): {len(search_space)}, {search_space}")
        
        search_space = SearchByProfile().filter_invalid_configs(args, search_space)
        print(f"[INFO] filter search_space: {len(search_space)}")

        SearchByProfile().search(get_args(), search_space)
    else:
        monitor_train_task()
