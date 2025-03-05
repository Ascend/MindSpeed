# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import datetime
import json
import math

import torch


SEARCH_CACHE_PATH = None
KV_STORE = None
PROFILE_CONTENT = {"fwd_time": [], "bwd_time": [], "act_mem": [], "module_param": []}
INITIAL_CONFIG = {}
GPT_ARGS_PATH = "gpt_args.json"
STAGE_PROFILE_PATH = 'stage_1_profile.json'


def broadcast_communicate(commum_data, source_rank):
    temp_data = torch.cuda.FloatTensor([commum_data])
    torch.distributed.broadcast(temp_data, src=source_rank)
    return temp_data.item()


def broadcast_communicate_list(commum_data, source_rank):
    temp_data = torch.cuda.FloatTensor(commum_data)
    torch.distributed.broadcast(temp_data, src=source_rank)
    return temp_data.tolist()


def cal_throughput(run_time, profile_data, parallel_cfg):
    sum_token = profile_data["text_decoder.seq_length"] * profile_data['grad_acc_step'] * profile_data['micro_batch_size']
    PP = parallel_cfg[0]
    TP = parallel_cfg[1]
    per_npu_throughput = sum_token / (run_time / 1000) / (PP * TP)
    return per_npu_throughput


def get_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def save_json(json_path, json_data):
    json_data_json = json.dumps(json_data)
    with open(json_path, 'w') as f:
        f.write(json_data_json)


def precise_round(num, ndigits=0):
    multiplier = 10 ** ndigits
    return math.floor(num * multiplier + 0.5) / multiplier
