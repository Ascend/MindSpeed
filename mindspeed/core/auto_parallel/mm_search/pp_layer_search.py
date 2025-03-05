# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import time
import functools
import operator

import numpy as np
import pulp
from pulp import LpMinimize, LpProblem, LpVariable, lpDot, lpSum
import highspy

from mindspeed.core.auto_parallel.mm_search.help import precise_round
from mindspeed.core.auto_parallel import BAND_WIDTH_UNIDIRECTIONAL


def get_send_recv_time(shape):
    data_size = functools.reduce(operator.mul, shape) * 2 / (1024 ** 3)
    return (data_size / BAND_WIDTH_UNIDIRECTIONAL) * 1e3


def pp_layer_search(parallel_cfg, profile_data, npu_memory_limit, last_stage_max_layer):
    print(f"[INFO] start pp layer search {time.ctime()}")
    print(f"[INFO] profile: {profile_data}")
    
    PP = parallel_cfg[0]
    DP = parallel_cfg[2]

    num_vit = profile_data["image_encoder.vision_encoder.num_layers"]
    num_llm = profile_data["text_decoder.num_layers"]
    
    model_structure = [1, num_vit - 2, 1, 1, num_llm - 2, 1]
    recomputing_fwd = [profile_data['vit']['fwd_time'], profile_data['llm']['fwd_time']]
    recomputing_act = [profile_data['vit']['act_mem'], profile_data['llm']['act_mem']]
    layer_name = ['vit_pre', 'vit', 'vit_post', 'llm_pre', 'llm', 'llm_post']
    model_num_layers = num_vit + num_llm
    print(f"PP:{PP}, DP:{DP}, num_vit, {num_vit}, num_llm, {num_llm}, model_num_layers, {model_num_layers}, \
          model_structure, {model_structure}")
    
    fwd_time, bwd_time, act_memory, static_memory = [], [], [], []
    for key in layer_name:
        fwd_time.append(int(profile_data[key]['fwd_time']))
        bwd_time.append(int(profile_data[key]['bwd_time']))
        act_memory.append(int(profile_data[key]['act_mem']))
        static_memory.append(int(sum(profile_data[key]['module_param'])))
    print(f"fwd_time, {fwd_time}, bwd_time, {bwd_time}, act_memory, {act_memory}, static_memory, {static_memory}")

    fwd_duration_layers, bwd_duration_layers, act_memory_layers, static_memory_layers = [], [], [], []
    for ind, num in enumerate(model_structure):
        fwd_duration_layers += num * [fwd_time[ind]]
        bwd_duration_layers += num * [bwd_time[ind]]
        act_memory_layers += num * [act_memory[ind]]
        static_memory_layers += num * [static_memory[ind]]

    memory_reserved = [npu_memory_limit] * PP
    num_micro_batches = profile_data["grad_acc_step"]
    if num_micro_batches < PP:
        return None, None, None
    
    send_recv_time = get_send_recv_time(
        [profile_data["text_decoder.seq_length"], profile_data["micro_batch_size"], profile_data["text_decoder.hidden_size"]]
    )
    comm_matrix = [[send_recv_time] * PP for _ in range(PP)]
    for i in range(PP):
        comm_matrix[i][i] = 0

    prob = LpProblem("Min_duration_time", LpMinimize)

    layer_placement = [LpVariable.matrix(f"X_{i}", range(model_num_layers), cat="Binary") for i in range(PP - 1)]
    
    # variable: forward/backward stage start time
    bwd_start, fwd_start = [], []
    for j in range(PP):
        fwd_start.append(LpVariable.matrix(f"fs_{j}", range(num_micro_batches), lowBound=1e-4, cat="Continuous"))
        bwd_start.append(LpVariable.matrix(f"bs_{j}", range(num_micro_batches), lowBound=1e-4, cat="Continuous"))
    recomputing_layers = [LpVariable.matrix("vit_r", range(PP - 1), lowBound=0, cat='Integer'),
                          LpVariable.matrix("llm_r", range(PP - 1), lowBound=0, cat='Integer')]

    layers_per_stage = [lpSum(layer_placement[s][i] for i in range(model_num_layers)) for s in range(PP - 1)]
    layers_per_stage.append(model_num_layers)

    Const1 = 0.0001
    Const2 = 10000
    Z = [LpVariable.matrix(f"Z_{i}", range(PP), cat="Binary") for i in range(2)]

    prob += recomputing_layers[0][0] + recomputing_layers[1][0] <= layers_per_stage[0]
    for s in range(1, PP - 1):
        prob += recomputing_layers[0][s] + recomputing_layers[1][s] <= layers_per_stage[s] - layers_per_stage[s - 1]
    for s in range(PP - 1):
        # constraint: llm recompute 
        prob += Z[1][s] <= 1 - (layers_per_stage[s] - num_vit) * Const1
        prob += Z[1][s] >= Const1 * (num_vit - layers_per_stage[s])
        prob += recomputing_layers[1][s] <= layers_per_stage[s] - num_vit + Const2 * Z[1][s]
        prob += recomputing_layers[1][s] <= Const2 * (1 - Z[1][s])
    prob += recomputing_layers[0][0] <= num_vit
    for s in range(1, PP - 1):
        # constraint: vit recompute 
        prob += Z[0][s] <= 1 - (layers_per_stage[s - 1] - num_vit) * Const1
        prob += Z[0][s] >= Const1 * (num_vit - layers_per_stage[s - 1])
        prob += recomputing_layers[0][s] <= num_vit - layers_per_stage[s - 1] + Const2 * (1 - Z[0][s])
        prob += recomputing_layers[0][s] <= Const2 * Z[0][s]
    
    # variable: pp stage forward/backward time
    fwd_duration_each_stage = []
    bwd_duration_each_stage = []
    fwd_duration_each_stage.append(lpSum(lpDot(fwd_duration_layers, layer_placement[0])))
    bwd_duration_each_stage.append(lpSum(lpDot(bwd_duration_layers, layer_placement[0]))
                                + recomputing_layers[0][0] * recomputing_fwd[0]
                                + recomputing_layers[1][0] * recomputing_fwd[1])
    for s in range(1, PP - 1):
        fwd_duration_each_stage.append(lpSum(lpDot(fwd_duration_layers, layer_placement[s])) - 
            lpSum(lpDot(fwd_duration_layers, layer_placement[s - 1])))
        bwd_duration_each_stage.append(lpSum(lpDot(bwd_duration_layers, layer_placement[s]))
                                    - lpSum(lpDot(bwd_duration_layers, layer_placement[s - 1]))
                                    + recomputing_layers[0][s] * recomputing_fwd[0]
                                    + recomputing_layers[1][s] * recomputing_fwd[1])
    fwd_duration_each_stage.append(sum(fwd_duration_layers) - lpSum(lpDot(fwd_duration_layers, layer_placement[-1])))
    bwd_duration_each_stage.append(sum(bwd_duration_layers) - lpSum(lpDot(bwd_duration_layers, layer_placement[-1])))

    prob += bwd_duration_each_stage[0] >= 1e-4

    # constraint: pp schedules constraints
    # warm up
    for s in range(PP):
        for j in range(PP - s - 1):
            prob += fwd_start[s][j] + fwd_duration_each_stage[s] <= fwd_start[s][j + 1]
    # cool down
    for s in range(PP):
        for j in range(num_micro_batches + s - PP, num_micro_batches - 1):
            prob += bwd_start[s][j] + bwd_duration_each_stage[s] <= bwd_start[s][j + 1]

    for s in range(PP):
        for j in range(num_micro_batches - PP + s + 1):
            prob += fwd_start[s][j + PP - s - 1] + fwd_duration_each_stage[s] <= bwd_start[s][j]

    for s in range(PP):
        for j in range(num_micro_batches - PP + s):
            prob += bwd_start[s][j] + bwd_duration_each_stage[s] <= fwd_start[s][j + PP - s]

    for s in range(PP - 1):
        for j in range(num_micro_batches):
            prob += fwd_start[s + 1][j] >= fwd_start[s][j] + fwd_duration_each_stage[s] + comm_matrix[s][s + 1]
            prob += bwd_start[s + 1][j] + bwd_duration_each_stage[s + 1] + comm_matrix[s + 1][s] <= bwd_start[s][j]

    # constraint: model layer placement
    for s in range(PP - 1):
        for i in range(model_num_layers - 1):
            prob += layer_placement[s][i] >= layer_placement[s][i + 1]

    for s in range(PP - 2):
        prob += (lpSum(layer_placement[s + 1][j] for j in range(model_num_layers)) >=
                 lpSum(layer_placement[s][j] for j in range(model_num_layers)) + 1)

    # constraint: model memory
    prob += ((lpSum(lpDot(layer_placement[0], act_memory_layers)) - 
                recomputing_layers[0][0] * recomputing_act[0]
                - recomputing_layers[1][0] * recomputing_act[1]) * (PP - 1) +
                lpSum(lpDot(layer_placement[0], act_memory_layers)) +
                lpSum(lpDot(layer_placement[0], static_memory_layers)) <= memory_reserved[0])
    for s in range(1, PP - 1):
        prob += ((lpSum(lpDot(layer_placement[s], act_memory_layers))
                - lpSum(lpDot(layer_placement[s - 1], act_memory_layers))
                - recomputing_layers[0][s] * recomputing_act[0]
                - recomputing_layers[1][s] * recomputing_act[1]) * (PP - s - 1) +
                lpSum(lpDot(layer_placement[s], act_memory_layers))
                - lpSum(lpDot(layer_placement[s - 1], act_memory_layers)) +
                lpSum(lpDot(layer_placement[s], static_memory_layers))
                - lpSum(lpDot(layer_placement[s - 1], static_memory_layers)) <= memory_reserved[s])

    prob += layer_placement[0][0] == 1

    prob += lpSum(layer_placement[-1][i] for i in range(model_num_layers)) >= model_num_layers - last_stage_max_layer

    # object function
    obj = bwd_start[0][num_micro_batches - 1] + bwd_duration_each_stage[0]
    prob += obj
    prob.writeLP("pp_layers_prob.lp")

    print(f"[INFO] start solve {time.ctime()}")
    h = highspy.Highs()
    filename = 'pp_layers_prob.lp'
    h.readModel(filename)
    h.run()
    print(f"[INFO] finish solve {time.ctime()}, solve state {h.modelStatusToString(h.getModelStatus())}")
    
    if h.modelStatusToString(h.getModelStatus()) != "Optimal":
        return None, None, None

    layer_placement_values = [[0 for t in range(model_num_layers)] for s in range(PP - 1)]
    recompute_values = [[0 for z in range(PP - 1)] for j in range(2)]
    e2e_time = 0
    for i, val in enumerate(h.getSolution().col_value):
        for s in range(PP - 1):
            for t in range(model_num_layers):
                if h.getColByName(str(layer_placement[s][t]))[1] == i:
                    layer_placement_values[s][t] = precise_round(val)
                    break
        for j in range(2):
            for z in range(PP - 1):
                if h.getColByName(str(recomputing_layers[j][z]))[1] == i:
                    recompute_values[j][z] = precise_round(val)
                    break
        if h.getColByName(str(bwd_start[0][num_micro_batches - 1]))[1] == i:
            e2e_time += int(val)
        for m in range(model_num_layers):
            if h.getColByName(str(layer_placement[0][m]))[1] == i:
                e2e_time += val * bwd_duration_layers[m]
                break
        for time_id in range(2):
            if h.getColByName(str(recomputing_layers[j][0]))[1] == i:
                e2e_time += val * recomputing_fwd[time_id]
                break
    
    layer_placement_result = np.array(layer_placement_values).sum(axis=1)
    print(f"[INFO] result: layer recompute: {recompute_values}")
    print(f"[INFO] the layer placement: {layer_placement_result}")
    print(f"[INFO] e2e time: {e2e_time}")

    return layer_placement_result, recompute_values, e2e_time

