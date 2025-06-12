import time
import json
import numpy as np
import torch
import torch_npu
from megatron.training import get_args
from megatron.training import print_rank_0


class PipelineParallelParas:
    def __init__(self,
                 num_stages,
                 fwd_durations,
                 bwd_durations,
                 num_microbatches,
                 comm_matrix,
                 num_layers):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.fwd_durations = fwd_durations
        self.bwd_durations = bwd_durations
        self.comm_matrix = comm_matrix
        self.num_layers = num_layers


def time_model_nfmb(paras, stage_schedule):
    # 给定一个调度序列，计算端到端时间
    num_stages = paras.num_stages
    num_mb = paras.num_microbatches
    comm_matrix = paras.comm_matrix
    chunk_placement = list(range(num_stages)) + list(range(num_stages - 1, -1, -1))
    # Fwd Bwd执行顺序
    fwd_bwd_comp_order = ([f'F_{i}' for i in range(num_stages)] +
                          [f'B_{i}' for i in range(num_stages - 1, -1, -1)])
    chunk_stage_map = dict(zip(fwd_bwd_comp_order, chunk_placement))

    if isinstance(stage_schedule, dict):
        stage_list = []
        for s in range(num_stages):
            fb_list = stage_schedule[f"stage{s}"]
            stage_list.append([element[0]+f"_{s}-"+element[1:] for element in fb_list])
    else:
        stage_list = stage_schedule

    # 初始化
    fwd_bwd_list = ([f"F_{j}-{i}" for i in range(num_mb) for j in range(num_stages)]
                    + [f"B_{j}-{i}" for i in range(num_mb) for j in range(num_stages)])
    values = [0 for _ in range(num_stages * num_mb * 2)]
    start_time = dict(zip(fwd_bwd_list, values))
    fwd_bwd_durations = dict()
    fwd_durations = np.array(paras.fwd_durations * num_mb).reshape(num_mb, num_stages).transpose()
    bwd_durations = np.array(paras.bwd_durations * num_mb).reshape(num_mb, num_stages).transpose()
    for j in range(num_stages):
        for i in range(num_mb):
            fwd_bwd_durations[f"F_{j}-{i}"] = fwd_durations[j, i]
            fwd_bwd_durations[f"B_{j}-{i}"] = bwd_durations[j, i]

    start_time[f"F_{0}-{0}"] = 0.1
    for s in range(num_stages - 1):
        start_time[f"F_{s + 1}-{0}"] = start_time[f"F_{s}-{0}"] + fwd_durations[s, 0] + comm_matrix[s][s + 1]

    # 获取当前任务的上一个任务以及依赖任务的结束时间
    def get_prev_task_time(task_start_time, task_list, pp_stage_id, mb_idx,
                          chunk_stage_map, comp_order, model_chunk_times,
                          comm_time_matrix):
        current_task = task_list[pp_stage_id][mb_idx]
        prev_task_same_stage = task_list[pp_stage_id][mb_idx - 1]
        chunk_id_prev_task_same_stage, _ = prev_task_same_stage.split('-')
        stage_id_prev_task = chunk_stage_map[chunk_id_prev_task_same_stage]
        chunk_position = comp_order.index(chunk_id_prev_task_same_stage)
        # 前一个任务计算完成后的通信时间
        if chunk_position < len(comp_order) - 1:
            stage_id_next = chunk_stage_map[comp_order[chunk_position + 1]]
            comm_time = comm_time_matrix[stage_id_prev_task][stage_id_next]
        else:
            comm_time = 0.01
        # 同一个stage上，前一个任务完成时间
        end_time_prev_task_stage = (task_start_time[prev_task_same_stage]
                                    + model_chunk_times[prev_task_same_stage]
                                    + comm_time)

        # 相同micro batch id，上一个model chunk上的计算时间
        cur_model_chunk, cur_mb = current_task.split('-')
        chunk_position = comp_order.index(cur_model_chunk)
        if chunk_position > 0:
            prev_model_chunk = comp_order[chunk_position - 1]
            prev_task_batch = prev_model_chunk + '-' + cur_mb
            comm_time = comm_time_matrix[chunk_stage_map[prev_model_chunk]][chunk_stage_map[cur_model_chunk]]
            end_time_dependent_task_batch = (task_start_time[prev_task_batch]
                                             + model_chunk_times[prev_task_batch]
                                             + comm_time)
            completed_flag = task_start_time[prev_task_same_stage] > 0 and task_start_time[prev_task_batch] > 0
        else:
            end_time_dependent_task_batch = 0.1
            completed_flag = task_start_time[prev_task_same_stage] > 0

        return end_time_prev_task_stage, end_time_dependent_task_batch, completed_flag

    # 更新计算时间
    begin_up = [1] * num_stages
    remaining = [num_mb * 2 - begin_up[p] for p in range(num_stages)]
    remaining_flag = True
    count = 0
    while remaining_flag:
        ids_old = []
        ids_new = []
        for s in range(num_stages):
            ids_old.append(remaining[s])
            if remaining[s]:
                microbatch_idx = len(stage_list[0]) - remaining[s]
                (end_time_prev_task_same_stage,
                 end_time_dependent_task_same_microbatch,
                 job_flag) = get_prev_task_time(start_time, stage_list, s, microbatch_idx, chunk_stage_map,
                                                fwd_bwd_comp_order, fwd_bwd_durations, comm_matrix)

                if job_flag:
                    start_time[stage_list[s][microbatch_idx]] = max(end_time_prev_task_same_stage,
                                                                    end_time_dependent_task_same_microbatch)
                    remaining[s] = remaining[s] - 1

            ids_new.append(remaining[s])

            if all(item == 0 for item in remaining):
                remaining_flag = False

        if ids_old == ids_new:
            count += 1
            if count == 3:
                start_time[f'B_0-{num_mb - 1}'] = 1e7
                break

    e2e_time = start_time[f'B_0-{num_mb - 1}'] + bwd_durations[0, -1]
    stage_start_time = [[start_time[job_name] for job_name in stage_list[s]] for s in range(num_stages)]

    return e2e_time, stage_start_time


def get_schedule_1f1b(paras):
    # generate 1f1b schedule list
    num_stages = paras.num_stages
    num_microbatches = paras.num_microbatches
    computation_placement = list(range(num_stages)) + list(range(num_stages - 1, -1, -1))

    # Fwd Bwd执行顺序
    fwd_bwd_order = ([f'F_{i}' for i in range(num_stages)] +
                     [f'B_{i}' for i in range(num_stages - 1, -1, -1)])

    # 根据1F1B策略生成每个stage上的调度顺序
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

    def get_stage_schedule(all_jobs_array, comp_placement, num_stages):
        stage_list = []
        for s in range(num_stages):
            stage_chunk_id = [index for index, element in enumerate(comp_placement) if element == s]
            warmup = num_stages - s
            stage_s_list = get_stage_list(all_jobs_array[stage_chunk_id[0]],
                                          all_jobs_array[stage_chunk_id[1]],
                                          warmup - 1)
            stage_list.append(stage_s_list)
        return stage_list

    all_jobs = np.array([[s + f'-{i}' for i in range(num_microbatches)] for s in fwd_bwd_order])
    stage_list = get_stage_schedule(all_jobs, computation_placement, num_stages)
    stage_schedule_dict = dict()
    for s in range(paras.num_stages):
        stage_s_list = []
        for element in stage_list[s]:
            item1, item2 = element.split("-")
            stage_s_list.append(item1[0] + item2)
        stage_schedule_dict[f"stage{s}"] = stage_s_list
    return stage_schedule_dict


def get_schedule_eager1f1b(paras, num_forwards, layers_placement):
    # generate 1f1b schedule list
    num_stages = paras.num_stages
    num_microbatches = paras.num_microbatches
    # 将原始模型切分为多个model chunk，chunk在PP stage上的放置顺序
    chunk_placement = list(range(num_stages)) + list(range(num_stages - 1, -1, -1))

    # Fwd Bwd执行顺序
    fwd_bwd_comp_order = ([f'F_{i}' for i in range(num_stages)] +
                          [f'B_{i}' for i in range(num_stages - 1, -1, -1)])

    # 根据1F1B策略生成每个stage上的调度顺序
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

    def get_stage_schedule(all_jobs_array, comp_placement, num_advanced, paras, layers_placement):
        stage_list = []
        activations_num = int(paras.num_layers // paras.num_stages) * (num_advanced + paras.num_stages)
        nums_under_memory = [int(activations_num // layers_placement[i]) for i in range(paras.num_stages)]
        warmups = [min(nums_under_memory[s] - s - 1,
                     2 * paras.num_stages - 2 * s - 2) for s in range(paras.num_stages)]
        for i in range(paras.num_stages - 1):
            warmups[i + 1] = min(warmups[i] - 1, warmups[i + 1])
            warmups[i + 1] = max(warmups[i + 1], 0)

        for s in range(paras.num_stages):
            stage_chunk_id = [index for index, element in enumerate(comp_placement) if element == s]
            num = sum(np.array(paras.bwd_durations[s + 1:])
                      + np.array(paras.fwd_durations[s + 1:])) // np.array(paras.fwd_durations[s])
            stage_s_list = get_stage_list(all_jobs_array[stage_chunk_id[0]],
                                          all_jobs_array[stage_chunk_id[1]],
                                          warmups[s])
            stage_list.append(stage_s_list)
        return stage_list

    all_jobs = np.array([[s + f'-{i}' for i in range(num_microbatches)] for s in fwd_bwd_comp_order])
    stage_list = get_stage_schedule(all_jobs, chunk_placement, num_forwards, paras, layers_placement)

    # 转换为dictionary
    stage_schedule_dict = dict()
    for s in range(paras.num_stages):
        stage_s_list = []
        for element in stage_list[s]:
            item1, item2 = element.split("-")
            stage_s_list.append(item1[0] + item2)
        stage_schedule_dict[f"stage{s}"] = stage_s_list

    return stage_schedule_dict


def schedule_layers(paras, num_mb_for_remaining_memory):
    # 调整层分布，对比层分布改变后，1F1B建模时间
    stage_layers = int(paras.num_layers // paras.num_stages)
    if paras.num_stages > 2:
        fwd_time_per_layer = sum(paras.fwd_durations[1:-1]) / (paras.num_stages - 2) / stage_layers
        bwd_time_per_layer = sum(paras.bwd_durations[1:-1]) / (paras.num_stages - 2) / stage_layers
    else:
        fwd_time_per_layer = paras.fwd_durations[0] / stage_layers
        bwd_time_per_layer = paras.bwd_durations[0] / stage_layers

    # 1f1b as baseline
    e2e_time = np.ones([2, paras.num_stages]) * 1e9
    paras_all = []
    layers_placement = []
    schedule_1f1b = get_schedule_1f1b(paras)
    e2e_time[0, 0], stage_start_time1 = time_model_nfmb(paras, schedule_1f1b)
    paras_all.append(paras)
    layers_p1 = [stage_layers] * paras.num_stages
    layers_placement.append(layers_p1)
    # 调度序列
    schedule_eager_1f1b = get_schedule_eager1f1b(paras, num_mb_for_remaining_memory, layers_p1)
    e2e_time[1, 0], stage_start_time2 = time_model_nfmb(paras, schedule_eager_1f1b)

    if stage_layers >= 2:
        for i in range(paras.num_stages - 1):
            fwd_new = np.array(paras.fwd_durations)
            fwd_new[i] += fwd_time_per_layer
            fwd_new[-1] -= fwd_time_per_layer
            bwd_new = np.array(paras.bwd_durations)
            bwd_new[i] += bwd_time_per_layer
            bwd_new[-1] -= bwd_time_per_layer
            paras1 = PipelineParallelParas(paras.num_stages,
                                           fwd_new.tolist(),
                                           bwd_new.tolist(),
                                           paras.num_microbatches,
                                           paras.comm_matrix,
                                           paras.num_layers)
            e2e_time[0, i + 1], stage_start_time1 = time_model_nfmb(paras1, schedule_1f1b)
            paras_all.append(paras1)
            layers_p1 = [stage_layers] * paras.num_stages
            layers_p1[i] += 1
            layers_p1[-1] -= 1
            layers_placement.append(layers_p1)
            schedule_eager_1f1b = get_schedule_eager1f1b(paras1, num_mb_for_remaining_memory, layers_p1)
            e2e_time[1, i + 1], stage_start_time2 = time_model_nfmb(paras1, schedule_eager_1f1b)

    optimal_paras = paras_all[e2e_time[1, :].argmin()]
    optimal_layer = layers_placement[e2e_time[1, :].argmin()]
    schedule_scheme = get_schedule_eager1f1b(optimal_paras, num_mb_for_remaining_memory, optimal_layer)

    return schedule_scheme, optimal_layer, e2e_time[1, :].min()


def broadcast_enable_schedule_in_ranks(src_rank, policy):
    enable_schedule = [False]
    if torch.distributed.get_rank() == src_rank:
        enable_schedule = [policy]
    tmp_enable_schedule = torch.cuda.BoolTensor(enable_schedule)
    torch.distributed.broadcast(tmp_enable_schedule, src=src_rank)
    return tmp_enable_schedule.item()


def broadcast_scheduler_in_ranks(src_rank, policy):
    args = get_args()
    policy_str = json.dumps(policy)
    byte_tensor = torch.cuda.ByteTensor(list(policy_str.encode()))
    torch.distributed.broadcast(byte_tensor, src_rank)
    if torch.distributed.get_rank() != 0:
        received_byte_tensor = torch.cuda.ByteTensor([0] * len(byte_tensor))
    else:
        received_byte_tensor = byte_tensor.clone()
    torch.distributed.broadcast(received_byte_tensor, src_rank)
    received_policy_str = ''.join([chr(byte) for byte in received_byte_tensor.tolist()])
    received_policy_data = json.loads(received_policy_str)
    args.pp_schedule_list = received_policy_data
    return received_policy_data


def broadcast_layer_in_ranks(src_rank, policy):
    args = get_args()
    num_layer_list = args.pipeline_model_parallel_size * [0]
    if torch.distributed.get_rank() == 0:
        num_layer_list = policy
    tmp_layer_list = torch.cuda.IntTensor(num_layer_list)
    torch.distributed.broadcast(tmp_layer_list, src=src_rank)
    args.num_layer_list = tmp_layer_list.tolist()
    return tmp_layer_list.tolist()


def all_gather_time(args, gather_time):
    recv_gather_time_list = []
    world_size = torch.distributed.get_world_size()
    gather_time = torch.cuda.FloatTensor([gather_time])
    gathered_tensors = [torch.zeros_like(gather_time) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_tensors, gather_time)
    for rank, tensor in enumerate(gathered_tensors):
        pipeline_stage_rank = get_pipeline_stage_rank(world_size, args.pipeline_model_parallel_size, rank)
        recv_gather_time_list.append((pipeline_stage_rank, tensor.item()))
    return recv_gather_time_list


def average_time_by_rank(time_list):
    time_dict = {}
    for item in time_list:
        if item[0] not in time_dict:
            time_dict[item[0]] = item[1]
        else:
            time_dict[item[0]] += item[1]
            time_dict[item[0]] /= 2
    return time_dict


def get_pipeline_stage_rank(world_size, num_stages, global_rank):
    assert world_size % num_stages == 0, "World size must be divisible by the number of stages"
    assert global_rank < world_size, "Global rank must be less than world size"

    stage_size = world_size // num_stages
    return global_rank // stage_size


def solve_pipelineschedule(args, data_parallel_size, num_forwards_first_stage, forward_time_dict, backward_time_dict):
    pipeline_stages = args.pipeline_model_parallel_size
    forward_time_each_stage = [forward_time_dict[rank] for rank in forward_time_dict]
    backward_time_each_stage = [backward_time_dict[rank] for rank in backward_time_dict]
    comm_matrix = [[0.05] * pipeline_stages for _ in range(pipeline_stages)]
    num_micro_batches = args.global_batch_size // data_parallel_size // args.micro_batch_size
    num_layers = args.num_layers

    pp_paras = PipelineParallelParas(pipeline_stages,
                                     forward_time_each_stage,
                                     backward_time_each_stage,
                                     num_micro_batches,
                                     comm_matrix,
                                     num_layers)
    # FB schedule
    start_time = time.time()
    schedule_list, optimal_layers, schedule_time = schedule_layers(pp_paras, num_forwards_first_stage)
    end_time = time.time()
    return schedule_list, schedule_time, optimal_layers
