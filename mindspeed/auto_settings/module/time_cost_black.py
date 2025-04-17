"""
算子预估
"""
import math
import os
from dataclasses import dataclass

import numpy as np
import operator
import functools

from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.module.operator.operator import Operator
from mindspeed.auto_settings.module.communication.communication import Communication
from mindspeed.auto_settings.utils.utils import get_num_warmup_micro_batches, get_module_info, get_prof_dir, \
    get_black_prof_file
from mindspeed.auto_settings.config.search_config import SearchConfig


@dataclass
class PipelineParallelParas:
    num_stages: int
    vpp: int
    fwd_durations: float
    bwd_durations: float
    num_microbatches: int
    comm_matrix: list


class TimeCostBlack(object):
    number_samples = 100
    band_width_undirectional = 25

    def __init__(self):
        self.logger = get_logger("TimeCostBlack")
        self.operator = Operator()
        self.communication = Communication()

    def crop_config(self, config: SearchConfig):
        """
        config
        """
        config.pipeline_model_parallel_size = 1
        config.num_layers_per_virtual_pipeline_stage = config.num_layers // config.pp
        return config

    def get_module_time(self, config: SearchConfig, module_name, node_rank):
        tmp_config = self.crop_config(config)
        path = get_black_prof_file(tmp_config)
        module = get_module_info(path, module_name)
        fwd_time = module.get('time', float('inf')) * 1000  # us
        forward_step_time = get_module_info(path, 'forward_step_time') * 1000  # us
        backward_step_time = get_module_info(path, 'backward_step_time') * 1000  # us
        return fwd_time, fwd_time / forward_step_time * backward_step_time

    def get_chunks_time_module_level(self, config: SearchConfig):
        forward_time_each_chunk = []
        backward_time_each_chunk = []
        nnodes = get_system_config().nnodes
        num_chunks = config.pp * config.vpp

        if num_chunks == 1:
            # 仅有一个chunk时不会对模型切分
            fwd_time = 0
            bwd_time = 0
            # embedding
            embedding = self.get_module_time(config, 'embedding', 0)
            fwd_time += embedding[0]
            bwd_time += embedding[1]
            # transformer
            transformer = self.get_module_time(config, '0', 0)
            fwd_time += transformer[0] * config.num_layers_per_virtual_pipeline_stage
            bwd_time += transformer[1] * config.num_layers_per_virtual_pipeline_stage
            # final_layernorm
            final_norm = self.get_module_time(config, 'final_layernorm', 0)
            fwd_time += final_norm[0]
            bwd_time += final_norm[1]
            # output_layer
            output_layer = self.get_module_time(config, 'output_layer', 0)
            fwd_time += output_layer[0]
            bwd_time += output_layer[1]
            # loss
            loss = self.get_module_time(config, 'loss', 0)
            fwd_time += loss[0]
            bwd_time += loss[1]

            forward_time_each_chunk.append(fwd_time)
            backward_time_each_chunk.append(bwd_time)

            return forward_time_each_chunk, backward_time_each_chunk

    def get_iteration_time(self, config: SearchConfig):
        iteration_times = np.array([0 for _ in range(TimeCostBlack.number_samples)]).astype(np.float64)

        time_chunks = self.get_chunks_time_module_level(config)
        for i in range(self.number_samples):
            iteration_times[i], _ = self.pipeline_costmodel(config, time_chunks[0], time_chunks[1])

        return iteration_times

    def get_send_recv_time(self, shape):
        data_size = functools.reduce(operator.mul, shape) * 2 / (1024 ** 3)  # GB
        return (data_size / self.band_width_undirectional) * 1e6

    def pipeline_costmodel(self, config: SearchConfig, fwd_time_chunks, bwd_time_chunks):
        model_config = get_model_config()

        send_recv_time = self.get_send_recv_time(
            [model_config.seq_length, config.micro_batch_size, model_config.hidden_size]
        )

        comm_matrix = [[send_recv_time] * config.pipeline_model_parallel_size \
                       for _ in range(config.pipeline_model_parallel_size)]
        for i in range(config.pipeline_model_parallel_size):
            comm_matrix[i][i] = 0

        paras = PipelineParallelParas(
            num_stages=config.pipeline_model_parallel_size,
            vpp=config.vpp,
            fwd_durations=fwd_time_chunks,
            bwd_durations=bwd_time_chunks,
            num_microbatches=config.mbs,
            comm_matrix=comm_matrix
        )

        scheduler_1f1b = self.get_schedule_1f1b(paras)

        e2e_time_1f1b, stage_start_time = self.time_model_nfmb(paras, scheduler_1f1b)

        return e2e_time_1f1b, stage_start_time

    def get_schedule_1f1b(self, paras):
        # generate 1f1b schedule list
        pp_stages = paras.num_stages
        vpp = paras.vpp
        num_microbatches = paras.num_microbatches
        computation_placement = list(range(pp_stages * vpp)) + list(range(pp_stages * vpp - 1, -1, -1))

        # Fwd和Bwd执行顺序
        fwd_bwd_order = ([f'F_{i}' for i in range(pp_stages * vpp)] +
                         [f'B_{i}' for i in range(pp_stages * vpp - 1, -1, -1)])

        # 根据1F1B策略生成每个stage上的调度顺序
        def get_stage_list(fwd_seq, bwd_seq, num_advanced):
            stage_order = []
            n = len(fwd_seq)
            # 判断序列中micro batch数目是否少于warm-up所需数目
            num_advanced = min(n, num_advanced)
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

        def get_stage_schedule(all_jobs_array, comp_placement, num_stages, vpp):
            stage_job_list = []
            for s in range(num_stages):
                stage_chunk_id = [index for index, element in enumerate(comp_placement) if (element % num_stages) == s]

                # 计算warmup的micro batch的数目
                if vpp > 1:
                    warmup = num_stages * (vpp + 1) - 2 * (s + 1)
                else:
                    warmup = num_stages - s - 1

                fwds = all_jobs_array[stage_chunk_id[0:vpp]]
                fwd_list = np.concatenate([fwds[:, index:index + num_stages].flatten()
                                           for index in range(0, np.size(all_jobs_array, 1), num_stages)])
                bwds = all_jobs_array[stage_chunk_id[vpp:]]
                bwd_list = np.concatenate([bwds[:, index:index + num_stages].flatten()
                                           for index in range(0, np.size(all_jobs_array, 1), num_stages)])
                stage_s_list = get_stage_list(fwd_list, bwd_list, warmup)
                stage_job_list.append(stage_s_list)
            return stage_job_list

        all_jobs = np.array([[s + f'-{i}' for i in range(num_microbatches)] for s in fwd_bwd_order])
        stage_list = get_stage_schedule(all_jobs, computation_placement, pp_stages, vpp)
        return stage_list

    def time_model_nfmb(self, paras, stage_list):
        # 给定一个调度序列，计算端到端时间
        num_pp_stages = paras.num_stages
        num_mb = paras.num_microbatches
        comm_matrix = paras.comm_matrix
        vpp = paras.vpp
        # vpp chunk放置顺序
        chunk_placement = list(range(num_pp_stages)) * vpp + list(range(num_pp_stages - 1, -1, -1)) * vpp
        # Fwd和Bwd执行顺序
        fwd_bwd_comp_order = ([f'F_{i}' for i in range(num_pp_stages * vpp)] +
                              [f'B_{i}' for i in range(num_pp_stages * vpp - 1, -1, -1)])
        chunk_stage_map = dict(zip(fwd_bwd_comp_order, chunk_placement))

        # 初始化
        fwd_bwd_list = ([f"F_{j}-{i}" for i in range(num_mb) for j in range(num_pp_stages * vpp)]
                        + [f"B_{j}-{i}" for i in range(num_mb) for j in range(num_pp_stages * vpp)])
        values = [0 for _ in range(num_pp_stages * vpp * num_mb * 2)]
        start_time = dict(zip(fwd_bwd_list, values))
        fwd_bwd_durations = dict()
        for j in range(num_pp_stages * vpp):
            for i in range(num_mb):
                fwd_bwd_durations[f"F_{j}-{i}"] = paras.fwd_durations[j]
                fwd_bwd_durations[f"B_{j}-{i}"] = paras.bwd_durations[j]

        start_time[f"F_{0}-{0}"] = 0.1
        for s in range(num_pp_stages - 1):
            start_time[f"F_{s + 1}-{0}"] = start_time[f"F_{s}-{0}"] + paras.fwd_durations[s] + comm_matrix[s][s + 1]

        # 获取当前任务的上一个任务以及依赖的前序任务的结束时间
        def get_prev_task_time(task_start_time,
                               task_list,
                               pp_stage_id,
                               task_idx,
                               chunk_stage_map,
                               comp_order,
                               model_chunk_times,
                               comm_time_matrix):
            current_task = task_list[pp_stage_id][task_idx]
            previous_task = task_list[pp_stage_id][task_idx - 1]
            previous_task_name, _ = previous_task.split('-')
            stage_id_previous_task = chunk_stage_map[previous_task_name]
            chunk_position = comp_order.index(previous_task_name)
            # 前一个任务计算完成后的通信时间
            if chunk_position < len(comp_order) - 1:
                stage_id_next = chunk_stage_map[comp_order[chunk_position + 1]]
                comm_time = comm_time_matrix[stage_id_previous_task][stage_id_next]
            else:
                comm_time = 0.01
            # 同一个stage上，前一个任务完成时间
            end_time_previous_task = (task_start_time[previous_task]
                                      + model_chunk_times[previous_task]
                                      + comm_time)

            # 同一个micro batch id，在前一个model chunk上依赖任务的计算时间
            chunk_name, cur_mb_index = current_task.split('-')
            chunk_position = comp_order.index(chunk_name)
            if chunk_position > 0:
                previous_chunk = comp_order[chunk_position - 1]
                dependent_task = previous_chunk + '-' + cur_mb_index
                comm_time = comm_time_matrix[chunk_stage_map[previous_chunk]][chunk_stage_map[chunk_name]]
                end_time_dependent_task = (task_start_time[dependent_task]
                                           + model_chunk_times[dependent_task]
                                           + comm_time)
                completed_flag = task_start_time[previous_task] > 0 and task_start_time[dependent_task] > 0
            else:
                end_time_dependent_task = 0.1
                completed_flag = task_start_time[previous_task] > 0

            return end_time_previous_task, end_time_dependent_task, completed_flag

        # 更新计算时间
        begin_up = [1] * num_pp_stages
        remaining = [num_mb * vpp * 2 - begin_up[p] for p in range(num_pp_stages)]
        remaining_flag = True
        count = 0
        while remaining_flag:
            ids_old = []
            ids_new = []
            for s in range(num_pp_stages):
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
                    self.logger.info("stage list is locked")
                    start_time[f'B_0-{num_mb - 1}'] = 1e7
                    break

        e2e_time = start_time[f'B_0-{num_mb - 1}'] + paras.bwd_durations[-1]
        stage_start_time = [[start_time[job_name] for job_name in stage_list[s]] for s in range(num_pp_stages)]

        return e2e_time, stage_start_time
