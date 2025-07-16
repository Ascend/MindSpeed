import os
import json
import math
import time
from datetime import datetime
from itertools import product
import numpy as np
import torch
from megatron.training import get_args
from megatron.training.arguments import parse_args
from mindspeed.arguments import parse_args_wrapper
from .autopipeline_perf import check_equal_model_configs


class Parallel_Paras:
    def __init__(self,
                 num_stages,
                 fwd_durations,
                 bwd_durations,
                 num_microbatch,
                 comm_matrix):
        self.num_stages = num_stages
        self.num_microbatch = num_microbatch
        self.fwd_durations = fwd_durations
        self.bwd_durations = bwd_durations
        self.comm_matrix = comm_matrix


def dynamic_mbs_1f1b(paras):
    num_stages = paras.num_stages
    num_microbatch = paras.num_microbatch
    computation_placement = list(range(num_stages)) + list(range(num_stages - 1, -1, -1))
    fwd_durations = paras.fwd_durations
    bwd_durations = paras.bwd_durations
    comm_matrix = paras.comm_matrix

    fwd_bwd_order = ([f'F_{i}' for i in range(num_stages)] +
                     [f'B_{i}' for i in range(num_stages - 1, -1, -1)])
    fwd_bwd_chunk_stage = dict(zip(fwd_bwd_order, computation_placement))

    def get_stage_list(fwd_seq, bwd_seq, num_advanced):
        stage_order = []
        n = len(fwd_seq)
        for idx in range(n):
            if idx < num_advanced:
                stage_order.append(fwd_seq[idx])
            else:
                stage_order.append(fwd_seq[idx])
                stage_order.append(bwd_seq[idx - num_advanced])
            if idx == n - 1:
                for i in range(num_advanced):
                    stage_order.append(bwd_seq[i - num_advanced])

        return stage_order

    def get_stage_schedule(all_jobs_array, comp_placement):
        stage_list = []
        for s in range(num_stages):
            stage_chunk_id = [index for index, element in enumerate(comp_placement) if element == s]
            warmup = num_stages - s
            stage_s_list = get_stage_list(all_jobs_array[stage_chunk_id[0]],
                                          all_jobs_array[stage_chunk_id[1]],
                                          warmup - 1)
            stage_list.append(stage_s_list)

        return stage_list

    all_jobs = np.array([[s + f'-{i}' for i in range(num_microbatch)] for s in fwd_bwd_order])
    stage_list = get_stage_schedule(all_jobs, computation_placement)

    fwd_bwd_list = ([f"F_{j}-{i}" for i in range(num_microbatch) for j in range(num_stages)]
                    + [f"B_{j}-{i}" for i in range(num_microbatch) for j in range(num_stages)])
    values = [0 for _ in range(num_stages * num_microbatch * 2)]
    start_time = dict(zip(fwd_bwd_list, values))
    fwd_bwd_durations = dict()
    for j in range(num_stages):
        for i in range(num_microbatch):
            fwd_bwd_durations[f"F_{j}-{i}"] = fwd_durations[j, i]
            fwd_bwd_durations[f"B_{j}-{i}"] = bwd_durations[j, i]

    for n in range(num_stages - 1):
        for s in range(n + 1):
            start_time[f"F_{s}-{n - s + 1}"] = max(start_time[f"F_{s}-{n - s + 1}"],
                                            start_time[f"F_{s}-{n - s}"] + fwd_durations[s, n - s] + comm_matrix[s][s + 1])
            start_time[f"F_{s + 1}-{n - s}"] = max(start_time[f"F_{s + 1}-{n - s}"],
                                             start_time[f"F_{s}-{n - s}"] + fwd_durations[s, n - s] + comm_matrix[s][s + 1])

    def get_prev_job_time(comp_start_time, pp_list, pp_id, mb_idx,
                          comp_chunk_stage, comp_order, model_chunk_times,
                          comm_time_matrix):
        current_job = pp_list[pp_id][mb_idx]
        prev_job_stage = pp_list[pp_id][mb_idx - 1]
        chunk_prev_job_stage, _ = prev_job_stage.split('-')
        stage_id_prev_job = comp_chunk_stage[chunk_prev_job_stage]
        chunk_position = comp_order.index(chunk_prev_job_stage)
        if chunk_position < len(comp_order) - 1:
            stage_id_next = comp_chunk_stage[comp_order[chunk_position + 1]]
            comm_time = comm_time_matrix[stage_id_prev_job][stage_id_next]
        else:
            comm_time = 0
        end_time_prev_job_stage = (comp_start_time[prev_job_stage] + model_chunk_times[prev_job_stage]
                                   + comm_time)

        cur_model_chunk, cur_mb = current_job.split('-')
        chunk_position = comp_order.index(cur_model_chunk)
        if chunk_position > 0:
            prev_model_chunk = comp_order[chunk_position - 1]
            prev_job_batch = prev_model_chunk + '-' + cur_mb
            comm_time = comm_time_matrix[comp_chunk_stage[prev_model_chunk]][comp_chunk_stage[cur_model_chunk]]
            end_time_prev_job_batch = comp_start_time[prev_job_batch] + model_chunk_times[prev_job_batch] + comm_time
            completed_flag = comp_start_time[prev_job_stage] > 0 and comp_start_time[prev_job_batch] > 0
        else:
            end_time_prev_job_batch = 0
            completed_flag = comp_start_time[prev_job_stage] > 0

        return end_time_prev_job_stage, end_time_prev_job_batch, completed_flag

    begin_up = [num_stages - s for s in range(num_stages)]
    remaining = [num_microbatch * 2 - begin_up[p] for p in range(num_stages)]
    remaining_flag = True
    while remaining_flag:
        ids_old = []
        ids_new = []
        for s in range(num_stages):
            ids_old.append(remaining[s])
            if remaining[s]:
                idx = len(stage_list[0]) - remaining[s]
                end_time_prev_stage, end_time_prev_batch, job_flag = get_prev_job_time(start_time, stage_list, s, idx,
                                                                                       fwd_bwd_chunk_stage,
                                                                                       fwd_bwd_order,
                                                                                       fwd_bwd_durations,
                                                                                       comm_matrix)

                if job_flag:
                    start_time[stage_list[s][idx]] = max(end_time_prev_stage, end_time_prev_batch)
                    remaining[s] = remaining[s] - 1

            ids_new.append(remaining[s])
            if all(item == 0 for item in remaining):
                remaining_flag = False
        if ids_old == ids_new:
            break

    e2e_time = start_time[f'B_0-{num_microbatch-1}'] + bwd_durations[0, -1]
    stage_start_time = [[start_time[job_name] for job_name in stage_list[s]] for s in range(num_stages)]
    return e2e_time, stage_start_time, stage_list, start_time


def find_integer_solutions(coefficients, global_batch_size):
    n = len(coefficients)
    mbs_max_value = (n + 1) // 2
    solutions = []
    all_comb = []
    for i in range(n):
        if i == mbs_max_value - 1:
            batch_using = sum(coefficients[0:mbs_max_value - 1] * 4)
            all_comb.append(list(range((global_batch_size - batch_using) // mbs_max_value,
                                       global_batch_size // mbs_max_value + 1)))
        else:
            all_comb.append(list(range(4)))

    for x in product(*all_comb):
        if sum(coefficients[i] * x[i] for i in range(n)) == global_batch_size:
            solutions.append(x)

    return solutions


def dynamic_mbs_search(num_stages, global_batch_size, fwd_mbs, bwd_mbs, comm_matrix):
    comp_mbs_ratio = [value / (index + 1) for index, value in enumerate(fwd_mbs)]
    fwd_mbs_selected = fwd_mbs[0:comp_mbs_ratio.index(min(comp_mbs_ratio)) + 1]
    bwd_mbs_selected = bwd_mbs[0:comp_mbs_ratio.index(min(comp_mbs_ratio)) + 1]
    mbs_max_value = len(fwd_mbs_selected)
    bwd_mbs_stages = [fwd_mbs_selected] * num_stages
    fwd_mbs_stages = [bwd_mbs_selected] * num_stages

    coefficients = list(range(1, mbs_max_value + 1)) + list(range(mbs_max_value - 1, 0, -1))
    solutions = find_integer_solutions(coefficients, global_batch_size)

    mbs_list = sum([solutions[0][i] * [coefficients[i]] for i in range(len(solutions[0]))], [])
    num_microbatch = len(mbs_list)
    fwd_durations = np.zeros([num_stages, num_microbatch])
    bwd_durations = np.zeros([num_stages, num_microbatch])
    for j in range(num_microbatch):
        for i in range(num_stages):
            fwd_durations[i, j] = fwd_mbs_stages[i][mbs_list[j] - 1]
            bwd_durations[i, j] = bwd_mbs_stages[i][mbs_list[j] - 1]

    paras = Parallel_Paras(num_stages, fwd_durations, bwd_durations, num_microbatch, comm_matrix)
    e2e_time = []
    for sol in solutions:
        mbs_list = sum([sol[i] * [coefficients[i]] for i in range(len(sol))], [])
        num_microbatch = len(mbs_list)
        fwd_durations = np.zeros([num_stages, num_microbatch])
        bwd_durations = np.zeros([num_stages, num_microbatch])
        for j in range(num_microbatch):
            for i in range(num_stages):
                fwd_durations[i, j] = fwd_mbs_stages[i][mbs_list[j] - 1]
                bwd_durations[i, j] = bwd_mbs_stages[i][mbs_list[j] - 1]

        paras.fwd_durations = fwd_durations
        paras.bwd_durations = bwd_durations
        paras.num_microbatch = num_microbatch

        e2e_time0, stage_start_time0, stage_list0, start_time0 = dynamic_mbs_1f1b(paras)
        e2e_time.append(e2e_time0)

    e2e_time_array = np.array(e2e_time)
    optimal_solution = solutions[e2e_time_array.argmin()]
    return optimal_solution, e2e_time_array.min()


def broadcast_oom_in_ranks(src_rank, policy):
    is_oom = [True]
    if torch.distributed.get_rank() == src_rank:
        is_oom = [policy]
    tmp_is_oom = torch.cuda.BoolTensor(is_oom)
    torch.distributed.broadcast(tmp_is_oom, src=src_rank)
    return tmp_is_oom.item()


def broadcast_mbs_in_ranks(src_rank, optimal_solution):
    args = get_args()
    solution_length = [0]
    if torch.distributed.get_rank() == src_rank:
        solution_length = [len(optimal_solution)]
    tmp_solution_length = torch.cuda.IntTensor(solution_length)
    torch.distributed.broadcast(tmp_solution_length, src=src_rank)
    solution_length = tmp_solution_length.item()

    tmp_optimal_solution = [0] * solution_length
    if torch.distributed.get_rank() == src_rank:
        tmp_optimal_solution = optimal_solution
    tmp_optimal_solution = torch.cuda.IntTensor(tmp_optimal_solution)
    torch.distributed.broadcast(tmp_optimal_solution, src=src_rank)
    tmp_optimal_solution = tmp_optimal_solution.tolist()
    mbs_max_value = math.ceil(len(tmp_optimal_solution) / 2)
    coefficients = list(range(1, mbs_max_value + 1)) + list(range(mbs_max_value - 1, 0, -1))
    optimal_mbs_list = sum([tmp_optimal_solution[i] * [coefficients[i]] for i in range(len(tmp_optimal_solution))], [])
    args.optimized_mbs_list = optimal_mbs_list
    return optimal_mbs_list


def get_profiling_data(policy, args):
    instance = {"model_configs": {
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "seq_length": args.seq_length,
        "num_attention_heads": args.num_attention_heads
    }, "optimpipeline_policy": [{
        "num_layers": args.num_layers,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "enable_scheduler": policy[0],
        "optimized_mbs_list": policy[1],
        "pp_schedule_list": policy[2],
        "optimal_layers": policy[3]
    }]}
    return instance


def save_profiling_data(policy, config_file):
    if torch.distributed.get_rank() % int(os.getenv('GPUS_PER_NODE', '8')) == 0:
        new_parse_args = parse_args_wrapper(parse_args)
        args = new_parse_args(None, False)
        instance = get_profiling_data(policy, args)
        if os.path.exists(config_file):
            with open(config_file, "r") as config_json:
                config_contents = config_json.read()
            parsed_contents = json.loads(config_contents)
            index = check_equal_model_configs(args, parsed_contents)
            if index != -1:
                if "optimpipeline_policy" in parsed_contents[index]:
                    parsed_contents[index]["optimpipeline_policy"].append(instance["optimpipeline_policy"][0])
            else:
                parsed_contents.append(instance)
            with open(config_file, "w") as f:
                json.dump(parsed_contents, f, ensure_ascii=False)
                os.chmod(config_file, 0o640)
        else:
            with open(config_file, "w") as f:
                json.dump([instance], f, ensure_ascii=False)
                os.chmod(config_file, 0o640)


def solve_optimpipeline(args, data_parallel_size, global_context):
    mbs_max_value = len(global_context)
    coefficients = list(range(1, mbs_max_value + 1)) + list(range(mbs_max_value - 1, 0, -1))
    optimal_solution = [0] * len(coefficients)
    optimal_time = 0
    if torch.distributed.get_rank() == 0:
        num_stages = args.pipeline_model_parallel_size
        global_batch_size = args.global_batch_size // data_parallel_size
        fwd_mbs = [item[0] for item in global_context]
        bwd_mbs = [item[1] for item in global_context]
        comm_matrix = [[0.05] * num_stages for _ in range(num_stages)]
        for i in range(num_stages):
            comm_matrix[i][i] = 0

        optimal_solution, optimal_time = dynamic_mbs_search(num_stages, global_batch_size, fwd_mbs, bwd_mbs, comm_matrix)
    torch.distributed.barrier()
    return optimal_solution, optimal_time
