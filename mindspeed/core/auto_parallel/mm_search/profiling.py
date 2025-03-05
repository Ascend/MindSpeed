# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import sys
import time
import copy
import operator
import subprocess
import signal
import threading
import json

import torch
import torch_npu

from megatron.training.global_vars import set_args, get_args
from megatron.core import parallel_state
from mindspeed.core.auto_parallel import get_kv_store
from mindspeed.core.auto_parallel.mm_search.help import (
    broadcast_communicate_list, 
    get_json, 
    save_json, 
    INITIAL_CONFIG, 
    PROFILE_CONTENT, 
    STAGE_PROFILE_PATH)
from mindspeed.core.auto_parallel.mm_search.solver import record_train_config
from mindspeed.core.auto_parallel.auto_parallel_profiling import BaseLaunch


class DistributedPerformanceProfiler(BaseLaunch):
    def update_args(self, config):
        args = get_args()
        self.old_args = copy.deepcopy(args)

        args.pipeline_model_parallel_size = config[0]
        args.tensor_model_parallel_size = config[1]
        args.data_parallel_size = config[2]
        args.micro_batch_size = config[3]


    def launch_model(self, config, profile_module):
        def update_or_append_param(argv: list, key, value=None):
            if not value:
                argv.append(key)
                return

            if key in argv:
                argv[argv.index(key) + 1] = value
            else:
                argv.extend([key, value])

        def remove_param(argv: list, key, has_value=False):
            if key in argv:
                pos = argv.index(key)
                argv.pop(pos)
                if has_value:
                    argv.pop(pos)

        def monitor_exit(process):
            while True:
                exit_flag = get_kv_store().get("exit_flag")
                if int(exit_flag) == 1:
                    try:
                        process_group_id = os.getpgid(process.pid)
                        os.killpg(process_group_id, signal.SIGKILL)
                        break
                    except ProcessLookupError:
                        break
                time.sleep(60)

        args = get_args()
        argv: list = sys.argv[1:]
        update_or_append_param(argv, '--eval-iters', '0')
        update_or_append_param(argv, '--train-iters', '5')
        update_or_append_param(argv, '--pipeline-model-parallel-size', str(args.pipeline_model_parallel_size))
        update_or_append_param(argv, '--tensor-model-parallel-size', str(args.tensor_model_parallel_size))
        update_or_append_param(argv, '--micro-batch-size', str(args.micro_batch_size))
        update_or_append_param(argv, '--auto-parallel-profile')
        update_or_append_param(argv, '--profile-subgraph-seg')
        update_or_append_param(argv, '--enable-dummy-optimizer')
        remove_param(argv, '--auto-parallel-mm')
        if profile_module == 'profiling_stage_1':
            update_or_append_param(argv, '--profile-stage', '1')
        elif profile_module == 'profiling_stage_2':
            update_or_append_param(argv, '--profile-stage', '2')

        command = [
            'torchrun',
            '--nproc_per_node', str(args.nproc_per_node),
            '--nnodes', str(args.nnodes),
            '--node-rank', str(args.node_rank),
            '--master_addr', str(args.master_addr),
            '--master_port', str(args.master_port),
            str(sys.argv[0])
        ] + argv
        print(' '.join(map(str, command)), flush=True)

        get_kv_store().set("exit_flag", "0")
        process = subprocess.Popen(command, shell=False, preexec_fn=lambda: os.setpgrp())
        monitor_thread = threading.Thread(target=monitor_exit, args=(process,))
        monitor_thread.start()
        status_code = process.wait()
        get_kv_store().set("exit_flag", "1")
        torch.distributed.barrier()
        return status_code


    def launch(self, config, profile_module):
        self.update_args(config)
        args = get_args()
        if args.node_rank != 0:
            self.launch_model(config, profile_module)
            super().recover_args()
            return None

        buffer = config + [0]
        torch.distributed.broadcast(torch.tensor(buffer, dtype=torch.int), 0)
        status_code = self.launch_model(config, profile_module)
        super().recover_args()

        return status_code


def save_profile_data(args):
    global PROFILE_CONTENT
    profile_content_json = json.dumps(PROFILE_CONTENT)
    with open(f'model_profile_{torch.distributed.get_rank()}.json', 'w') as f:
        f.write(profile_content_json)
    if args.profile_subgraph_seg:
        PROFILE_CONTENT = get_profile_from_rank(args)
        PROFILE_CONTENT = record_train_config(PROFILE_CONTENT)

        if torch.distributed.get_rank() == 0:
            profile_content_json = json.dumps(PROFILE_CONTENT)
            with open(f'model_profile.json', 'w') as f:
                f.write(profile_content_json)
        print(PROFILE_CONTENT)


def set_profile_model_config(args):
    vit_model_args = ["num_layers"]
    llm_model_args = ["num_layers", "seq_length", "hidden_size"]
    train_args = ["micro_batch_size", "use_distributed_optimizer", "simulated_nproc_per_node", "simulated_nnodes"]
    for arg in vit_model_args:
        if hasattr(args.mm.model.image_encoder.vision_encoder, arg):
            INITIAL_CONFIG[f"image_encoder.vision_encoder.{arg}"] = getattr(args.mm.model.image_encoder.vision_encoder, arg)
    for arg in llm_model_args:
        if hasattr(args.mm.model.text_decoder, arg):
            INITIAL_CONFIG[f"text_decoder.{arg}"] = getattr(args.mm.model.text_decoder, arg)
    for arg in train_args:
        if hasattr(args, arg):
            INITIAL_CONFIG[arg] = getattr(args, arg)

    if args.profile_stage == 1:
        args.mm.model.image_encoder.vision_encoder.num_layers = 2
        args.mm.model.image_encoder.vision_encoder.pipeline_num_layers = [1, ] * 2 + [0, ] * 2
        args.mm.model.text_decoder.num_layers = 2
        args.mm.model.text_decoder.pipeline_num_layers = [0, ] * 2 + [1, ] * 2
    elif args.profile_stage == 2:
        args.mm.model.image_encoder.vision_encoder.num_layers = 4
        args.mm.model.image_encoder.vision_encoder.pipeline_num_layers = [2, ] * 2 + [0, ] * 2
        args.mm.model.text_decoder.num_layers = 4
        args.mm.model.text_decoder.pipeline_num_layers = [0, ] * 2 + [2, ] * 2

    recompute_args = ["recompute_granularity", "recompute_method", "recompute_num_layers"]
    for arg in recompute_args:
        if hasattr(args.mm.model.image_encoder.vision_encoder, arg):
            setattr(args.mm.model.image_encoder.vision_encoder, arg, None)
        if hasattr(args.mm.model.image_encoder.vision_projector, arg):
            setattr(args.mm.model.image_encoder.vision_projector, arg, None)
        if hasattr(args.mm.model.text_decoder, arg):
            setattr(args.mm.model.text_decoder, arg, None)

    print(f"[INFO] initial_config:", INITIAL_CONFIG)
    print(f"[INFO] finish: vit pp layer: {args.mm.model.image_encoder.vision_encoder.pipeline_num_layers}, \
        vit num layer: {args.mm.model.image_encoder.vision_encoder.num_layers}, \
        llm pp layer: {args.mm.model.text_decoder.pipeline_num_layers}, \
        llm num layer: {args.mm.model.text_decoder.num_layers}, \
        PP: {args.pipeline_model_parallel_size}, \
        TP: {args.tensor_model_parallel_size}")


def get_profile_from_rank(args):
    global PROFILE_CONTENT

    def get_average_time(data, m=2):
        data = sorted(data)
        median = data[len(data) // 2]
        normal = [x for x in data if median - m * median < x < median + m * median]
        try:
            average = sum(normal) / len(normal)
            return average
        except ZeroDivisionError:
            print("[Error] Divided by zero.")
            return None

    def get_computer_time():
        if "fwd_time" in PROFILE_CONTENT:
            PROFILE_CONTENT["fwd_time"] = get_average_time(PROFILE_CONTENT["fwd_time"])
        else:
            PROFILE_CONTENT["fwd_time"] = 0
        if "bwd_time" in PROFILE_CONTENT:
            PROFILE_CONTENT["bwd_time"] = get_average_time(PROFILE_CONTENT["bwd_time"])
        else:
            PROFILE_CONTENT["bwd_time"] = 0 
        if "act_mem" in PROFILE_CONTENT:
            PROFILE_CONTENT["act_mem"] = get_average_time(PROFILE_CONTENT["act_mem"])
        else:
            PROFILE_CONTENT["act_mem"] = 0 

    get_computer_time()
    
    tp_device_num = args.tensor_model_parallel_size
    pp_device_num = args.pipeline_model_parallel_size
    dp_device_num = int(args.nnodes * args.nproc_per_node / tp_device_num / pp_device_num)
    
    profile_data_list = []
    for rank_id in range(args.pipeline_model_parallel_size):
        fwd_time, bwd_time, model_mem, act_mem = 0, 0, [0, 0, 0], 0
        if parallel_state.get_pipeline_model_parallel_rank() == rank_id:
            fwd_time = PROFILE_CONTENT['fwd_time']
            bwd_time = PROFILE_CONTENT['bwd_time']
            model_mem = PROFILE_CONTENT['module_param']
            act_mem = PROFILE_CONTENT['act_mem']
        profile_rank_data = [fwd_time, bwd_time, act_mem] + model_mem
        profile_rank_data = broadcast_communicate_list(profile_rank_data, rank_id * tp_device_num * dp_device_num)
        profile_data_list.append(profile_rank_data)
    
    if args.profile_stage == 1:
        PROFILE_CONTENT = {}
        PROFILE_CONTENT['vit_pre'] = {"fwd_time": profile_data_list[0][0], 
                                        "bwd_time": profile_data_list[0][1], 
                                        "module_param": profile_data_list[0][-3:], 
                                        "act_mem": profile_data_list[0][2]}
        PROFILE_CONTENT['vit_post'] = {"fwd_time": profile_data_list[1][0], 
                                        "bwd_time": profile_data_list[1][1], 
                                        "module_param": profile_data_list[1][-3:], 
                                        "act_mem": profile_data_list[1][2]}
        PROFILE_CONTENT['llm_pre'] = {"fwd_time": profile_data_list[2][0], 
                                        "bwd_time": profile_data_list[2][1], 
                                        "module_param": profile_data_list[2][-3:], 
                                        "act_mem": profile_data_list[2][2]}
        PROFILE_CONTENT['llm_post'] = {"fwd_time": profile_data_list[3][0], 
                                        "bwd_time": profile_data_list[3][1], 
                                        "module_param": profile_data_list[3][-3:], 
                                        "act_mem": profile_data_list[3][2]}
        
        save_json(STAGE_PROFILE_PATH, PROFILE_CONTENT)

    elif args.profile_stage == 2:
        profile_data = get_json(STAGE_PROFILE_PATH)

        PROFILE_CONTENT = copy.deepcopy(profile_data)
        model_mem_vit = copy.deepcopy(profile_data_list[0][-3:])
        model_mem_llm = copy.deepcopy(profile_data_list[2][-3:])
        for i, v in enumerate(profile_data['vit_pre']['module_param']):
            model_mem_vit[i] = (profile_data_list[0][-3:][i] - profile_data['vit_pre']['module_param'][i])
            model_mem_llm[i] = (profile_data_list[2][-3:][i] - profile_data['llm_pre']['module_param'][i])

        PROFILE_CONTENT['vit'] = {"fwd_time": (profile_data_list[0][0] - profile_data['vit_pre']['fwd_time']), 
                                    "bwd_time": (profile_data_list[0][1] - profile_data['vit_pre']['bwd_time']), 
                                    "module_param": model_mem_vit, 
                                    "act_mem": (profile_data_list[0][2] - profile_data['vit_pre']['act_mem'])}
        PROFILE_CONTENT['llm'] = {"fwd_time": (profile_data_list[2][0] - profile_data['llm_pre']['fwd_time']), 
                                    "bwd_time": (profile_data_list[2][1] - profile_data['llm_pre']['bwd_time']), 
                                    "module_param": model_mem_llm, 
                                    "act_mem": (profile_data_list[2][2] - profile_data['llm_pre']['act_mem']), 
                                    "embed_time": profile_data['llm_pre']['fwd_time'] - (profile_data_list[2][0] - profile_data['llm_pre']['fwd_time']) if profile_data['llm_pre']['fwd_time'] - (profile_data_list[2][0] - profile_data['llm_pre']['fwd_time']) > 0 else 0}

    return PROFILE_CONTENT

