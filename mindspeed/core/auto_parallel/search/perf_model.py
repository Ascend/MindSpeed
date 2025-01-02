# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import time
import math
from functools import reduce
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch_npu

from .operator_model_cache import operator_cache
from .operator_model import Sampler
from ..utils import logger, get_model_config, get_system_config
from ..utils.parallel_config import ParallelConfig
from ..utils.utils import GlobalMemoryBuffer
from ..utils.utils import get_module_info
from ..utils.profiler import CommProfiling


_GLOBAL_ATTN_MASK = None
_ITERATION_LOOP_TIME = 5


def get_attention_mask():
    global _GLOBAL_ATTN_MASK
    args = get_model_config().args
    if args.use_flash_attn and (
        args.seq_length > 2048 or args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']
    ):
        args.sparse_mode = 2
        _GLOBAL_ATTN_MASK = torch.triu(
            torch.ones([2048, 2048], dtype=bool, device=torch.cuda.current_device()), diagonal=1
        )
    else:
        args.sparse_mode = 0
        _GLOBAL_ATTN_MASK = (
            torch.tril(
                torch.ones(
                    [args.micro_batch_size, 1, args.seq_length, args.seq_length], 
                    dtype=bool, 
                    device=torch.cuda.current_device()
                ), 
                diagonal=-(args.pre_tockens + 1)
            )
            + 
            torch.triu(
                torch.ones(
                    [args.micro_batch_size, 1, args.seq_length, args.seq_length], 
                    dtype=bool, 
                    device=torch.cuda.current_device()
                ), 
                diagonal=args.next_tockens + 1
            )
        )
    return _GLOBAL_ATTN_MASK


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, inputs):
        x, y = inputs
        return torch.matmul(x, y.t())


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps)

    def forward(self, x):
        return self.layer_norm(*x)


class FusedRmsNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float16)).npu()
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x[0], self.weight, epsilon=self.eps)[0]


class BatchMatMul(torch.nn.Module):
    def __init__(self):
        super(BatchMatMul, self).__init__()

    def forward(self, inputs):
        x, y = inputs
        return torch.bmm(x, y)


class FlashAttention(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.pre_tockens = 65536
        self.next_tockens = 0

        self.attention_mask = get_attention_mask()

    def forward(self, x):
        q, k, v = x
        seq_length, _, hd = q.shape[0], q.shape[1], q.shape[2]
        head_num = hd // self.head_dim
        output = torch_npu.npu_fusion_attention(
            q, k, v, head_num, 'SBH',
            pse=None,
            padding_mask=None,
            atten_mask=self.attention_mask,
            scale=self.scale,
            pre_tockens=self.pre_tockens,
            next_tockens=self.next_tockens,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=get_model_config().args.sparse_mode
        )[0]
        return output


class TransformerBlock:
    def __init__(self, config: ParallelConfig, number_sample):
        self.number_sample = number_sample
        self.config = config
        self.noise_model = OperatorNoiseSampler(self.number_sample)

    def norm(self):
        args = get_model_config().args
        tp = self.config.tensor_model_parallel_size
        cp = self.config.ring_attention_size
        up = self.config.ulysses_size
        mbs = self.config.micro_batch_size
        input_shape = [args.seq_length // cp // tp // up, mbs, args.hidden_size]
        if args.normalization == 'RMSNorm':
            ftime, btime = self.noise_model.fused_rms_norm(input_shape, input_shape, args.hidden_size)
        else:
            ftime, btime = self.noise_model.layernorm(input_shape, input_shape, args.hidden_size)
        return ftime, btime

    def self_attention_time(self):
        args = get_model_config().args
        tp = self.config.tensor_model_parallel_size
        cp = self.config.ring_attention_size
        up = self.config.ulysses_size
        mbs = self.config.micro_batch_size
        ftime, btime = self.noise_model.flash_attention(
            [args.seq_length // cp, mbs, args.hidden_size // tp // up],
            [args.seq_length // cp, mbs, args.hidden_size // tp // up],
            [args.seq_length // cp, mbs, args.hidden_size // tp // up],
            [args.seq_length // cp, mbs, args.hidden_size // tp // up],
            args.hidden_size // args.num_attention_heads,
        )
        return ftime, btime

    def parallel_attention_time(self):
        args = get_model_config().args
        s = args.seq_length
        a = args.num_attention_heads
        h = args.hidden_size
        d = args.hidden_size // a
        b = self.config.micro_batch_size
        tp = self.config.tensor_model_parallel_size
        cp = self.config.ring_attention_size
        up = self.config.ulysses_size
        head_split = a // tp
        
        fwd_time, bwd_time = 0, 0

        if args.group_query_attention:
            ng = args.num_query_groups // tp
            ftime, btime = self.noise_model.matmul(
                [s // cp // up * b, h],
                [ng * (head_split // ng + 2) * d, h],
                [s // cp // up * b, ng * (head_split // ng + 2) * d]
            )
        else:
            ftime, btime = self.noise_model.matmul(
                [s // cp // up * b, h],
                [3 * head_split * d, h],
                [s // cp // up * b, 3 * head_split * d]
            )
        fwd_time += ftime
        bwd_time += btime

        if not args.use_flash_attn:
            raise AssertionError('the auto-parallel only support FA')
        else:
            alltoall_time = CommProfiling.get_comm_time([s // cp // up, b, a // tp, d], up, 'alltoall')
            fwd_time += (3 * alltoall_time)
            bwd_time += (3 * alltoall_time)

            send_recv_time = CommProfiling.get_send_recv_time([2, 2, s // cp // 2, b, a // tp // up * d])
            ftime, btime = self.self_attention_time()
            for _ in range(cp - 1):
                fwd_time += max([ftime.max(), send_recv_time])
                bwd_time += max([btime.max(), send_recv_time])
            fwd_time += ftime
            bwd_time += btime

            alltoall_time = CommProfiling.get_comm_time([s // cp, b, a // tp // up, d], up, 'alltoall')
            fwd_time += alltoall_time
            bwd_time += alltoall_time

        ftime, btime = self.noise_model.matmul([s // cp // up * b, h // tp], [h, h // tp], [s // cp // up * b, h])
        fwd_time += ftime
        bwd_time += btime

        return fwd_time, bwd_time

    def get_block_time(self):
        args = get_model_config().args
        s = args.seq_length
        a = args.num_attention_heads
        h = args.hidden_size
        ffn = args.ffn_hidden_size
        d = args.hidden_size // args.num_attention_heads
        b = self.config.micro_batch_size
        tp = self.config.tensor_model_parallel_size
        cp = self.config.ring_attention_size
        up = self.config.ulysses_size

        fwd_time = np.array([0 for _ in range(self.number_sample)]).astype(np.float64)
        bwd_time = np.array([0 for _ in range(self.number_sample)]).astype(np.float64)

        ftime, btime = self.norm()
        fwd_time += ftime
        bwd_time += btime

        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        fwd_time += all_gather_time
        bwd_time += reduce_scatter_time

        ftime, btime = self.parallel_attention_time()
        fwd_time += ftime
        bwd_time += btime

        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        fwd_time += reduce_scatter_time
        bwd_time += all_gather_time

        ftime, btime = self.norm()
        fwd_time += ftime
        bwd_time += btime

        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        fwd_time += all_gather_time
        bwd_time += reduce_scatter_time

        ftime, btime = self.noise_model.matmul([s // cp // up * b, h], [ffn // tp, h], [s // cp // up * b, ffn // tp])
        fwd_time += ftime
        bwd_time += btime

        ftime, btime = self.noise_model.matmul([s // cp // up * b, ffn // tp], [h, ffn // tp], [s // cp // up * b, h])
        fwd_time += ftime
        bwd_time += btime

        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        fwd_time += reduce_scatter_time
        bwd_time += all_gather_time

        return fwd_time, bwd_time


class OperatorNoiseSampler:
    def __init__(self, num_sample=100):
        self.sampling = Sampler(num_sample=num_sample)

    @staticmethod
    def measure_matmul_time(left_shape, left_transpose, right_shape, right_transpose):
        left_matrix = GlobalMemoryBuffer.get_tensor(left_shape, 0)
        left_matrix = left_matrix if not left_transpose else left_matrix.t()
        right_matrix = GlobalMemoryBuffer.get_tensor(right_shape, 1)
        right_matrix = right_matrix if not right_transpose else right_matrix.t()

        for _ in range(_ITERATION_LOOP_TIME):
            torch.matmul(left_matrix, right_matrix)

        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(_ITERATION_LOOP_TIME):
            torch.matmul(left_matrix, right_matrix)
        torch.npu.synchronize()
        return (time.time() - start_time) * 1e6 / _ITERATION_LOOP_TIME

    @staticmethod
    def measure_batchmatmul_time(left_shape, left_transpose, right_shape, right_transpose):
        left_matrix = GlobalMemoryBuffer.get_tensor(left_shape, 0)
        left_matrix = left_matrix if not left_transpose else left_matrix.permute(0, 2, 1)
        right_matrix = GlobalMemoryBuffer.get_tensor(right_shape, 0)
        right_matrix = right_matrix if not right_transpose else right_matrix.permute(0, 2, 1)

        for _ in range(_ITERATION_LOOP_TIME):
            torch.bmm(left_matrix, right_matrix)

        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(_ITERATION_LOOP_TIME):
            torch.bmm(left_matrix, right_matrix)
        torch.npu.synchronize()
        return (time.time() - start_time) * 1e6 / _ITERATION_LOOP_TIME

    def matmul(self, input_shape1, input_shape2, output_shape):
        ftime, _, from_cache = operator_cache.find('MatMul', [input_shape1, input_shape2])
        if not from_cache:
            ftime = self.measure_matmul_time(input_shape1, False, input_shape2, True)
        ftime_uncertainty = self.sampling.run('MatMul', ftime, output_shape, input_shape1, input_shape2)
        operator_cache.record('MatMul', [input_shape1, input_shape2], output_shape, ftime, 0)

        btime1, _, from_cache = operator_cache.find('MatMul', [output_shape, input_shape2])
        if not from_cache:
            btime1 = self.measure_matmul_time(output_shape, False, input_shape2, False)
        btime1_uncertainty = self.sampling.run('MatMul', btime1, input_shape1, output_shape, input_shape2)
        operator_cache.record('MatMul', [output_shape, input_shape2], input_shape1, btime1, 0)

        btime2, _, from_cache = operator_cache.find('MatMul', [output_shape, input_shape1])
        if not from_cache:
            btime2 = self.measure_matmul_time(output_shape, True, input_shape1, False)
        btime2_uncertainty = self.sampling.run('MatMul', btime2, input_shape2, output_shape, input_shape1)
        operator_cache.record('MatMul', [output_shape, input_shape1], input_shape2, btime2, 0)
        return ftime_uncertainty, btime1_uncertainty + btime2_uncertainty

    def batch_matmul(self, input_shape1, input_shape2, output_shape):
        ftime, _, from_cache = operator_cache.find('BatchMatMul', [input_shape1, input_shape2])
        if not from_cache:
            ftime = self.measure_batchmatmul_time(input_shape1, False, input_shape2, False)
        ftime_uncertainty = self.sampling.run('BatchMatMul', ftime, output_shape, input_shape1, input_shape2)
        operator_cache.record('BatchMatMul', [input_shape1, input_shape2], output_shape, ftime, 0)

        btime1, _, from_cache = operator_cache.find('BatchMatMul', [input_shape1, output_shape])
        if not from_cache:
            btime1 = self.measure_batchmatmul_time(input_shape1, True, output_shape, False)
        btime1_uncertainty = self.sampling.run('BatchMatMul', btime1, input_shape2, input_shape1, output_shape)
        operator_cache.record('BatchMatMul', [input_shape1, output_shape], input_shape2, btime1, 0)

        btime2, _, from_cache = operator_cache.find('BatchMatMul', [output_shape, input_shape2])
        if not from_cache:
            btime2 = self.measure_batchmatmul_time(output_shape, False, input_shape2, True)
        btime2_uncertainty = self.sampling.run('BatchMatMul', btime2, input_shape1, output_shape, input_shape2)
        operator_cache.record('BatchMatMul', [output_shape, input_shape2], input_shape1, btime2, 0)
        return ftime_uncertainty, btime1_uncertainty + btime2_uncertainty

    def layernorm(self, input_shape, output_shape, hidden_size, eps=1e-5):
        layernorm = LayerNorm(hidden_size, eps)
        ftime, btime, from_cache = operator_cache.find('LayerNorm', input_shape)
        if not from_cache:
            ftime, btime = profile(layernorm, [input_shape])
        ftime_uncertainty = self.sampling.run('LayerNorm', ftime, output_shape, input_shape)
        btime_uncertainty = self.sampling.run('LayerNormGrad', btime, input_shape, output_shape)
        operator_cache.record('LayerNorm', input_shape, output_shape, ftime, btime)
        return ftime_uncertainty, btime_uncertainty

    def fused_rms_norm(self, input_shape, output_shape, hidden_size, eps=1e-6):
        fused_rms_norm = FusedRmsNorm(hidden_size, eps)
        ftime, btime, from_cache = operator_cache.find('RmsNorm', input_shape)
        if not from_cache:
            ftime, btime = profile(fused_rms_norm, [input_shape])
        ftime_uncertainty = self.sampling.run('RmsNorm', ftime, output_shape, input_shape)
        btime_uncertainty = self.sampling.run('RmsNormGrad', btime, output_shape, input_shape)
        operator_cache.record('RmsNorm', input_shape, output_shape, ftime, btime)
        return ftime_uncertainty, btime_uncertainty

    def flash_attention(self, q, k, v, output_shape, head_dim):
        flash_attn = FlashAttention(head_dim)
        ftime, btime, from_cache = operator_cache.find('FlashAttentionScore', [q, k, v])
        if not from_cache:
            ftime, btime = profile(flash_attn, [q, k, v])
        ftime_uncertainty = self.sampling.run('FlashAttentionScore', ftime, output_shape, q, k, v)
        btime_uncertainty = self.sampling.run('FlashAttentionScoreGrad', btime, output_shape, q, k, v)
        operator_cache.record('FlashAttentionScore', [q, k, v], q, ftime, btime)
        return ftime_uncertainty, btime_uncertainty


@dataclass
class PipelineParallelParas:
    num_stages: int
    vpp: int
    fwd_durations: float
    bwd_durations: float
    num_microbatches: int
    comm_matrix: list


def get_schedule_1f1b(paras):
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


def time_model_nfmb(paras, stage_list):
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
    def get_prev_task_time(prev_task_params_group):
        task_start_time = prev_task_params_group[0]
        task_list = prev_task_params_group[1]
        pp_stage_id = prev_task_params_group[2]
        task_idx = prev_task_params_group[3]
        chunk_stage_map = prev_task_params_group[4]
        comp_order = prev_task_params_group[5]
        model_chunk_times = prev_task_params_group[6]
        comm_time_matrix = prev_task_params_group[7]

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
                params_group = (start_time, stage_list, s, microbatch_idx, chunk_stage_map,
                                fwd_bwd_comp_order, fwd_bwd_durations, comm_matrix)
                (end_time_prev_task_same_stage,
                 end_time_dependent_task_same_microbatch,
                 job_flag) = get_prev_task_time(params_group)

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
                logger.info("stage list is locked")
                start_time[f'B_0-{num_mb - 1}'] = 1e7
                break

    e2e_time = start_time[f'B_0-{num_mb - 1}'] + paras.bwd_durations[-1]
    stage_start_time = [[start_time[job_name] for job_name in stage_list[s]] for s in range(num_pp_stages)]

    return e2e_time, stage_start_time


def profile(model, shapes):
    model.to(torch.cuda.current_device())

    input_tensors = []
    index = 0
    for shape in shapes:
        tensor = GlobalMemoryBuffer.get_tensor(shape, index).requires_grad_()
        input_tensors.append(tensor)
        index += 1

    sum_z = None
    for _ in range(3):
        sum_z = model(input_tensors)

    # forward_time
    torch.npu.synchronize()
    start_time = time.time()
    for _ in range(_ITERATION_LOOP_TIME):
        model(input_tensors)
    torch.npu.synchronize()
    fwd_time = (time.time() - start_time) * 1e6 / _ITERATION_LOOP_TIME

    for _ in range(3):
        z = model(input_tensors)
        loss = torch.sum(z)
        loss.backward()

    torch.npu.synchronize()
    start_time = time.time()
    for _ in range(_ITERATION_LOOP_TIME):
        torch.sum(sum_z)
    torch.npu.synchronize()
    loss_time = (time.time() - start_time) * 1e6 / _ITERATION_LOOP_TIME

    torch.npu.synchronize()
    start_time = time.time()
    for _ in range(_ITERATION_LOOP_TIME):
        z = model(input_tensors)
        loss = torch.sum(z)
        loss.backward()
    torch.npu.synchronize()
    bwd_time = (time.time() - start_time) * 1e6 / _ITERATION_LOOP_TIME - fwd_time - loss_time
    return fwd_time, bwd_time


class TimeCostModel:
    number_samples = 100

    @staticmethod
    def get_module_time(config: ParallelConfig, module_name, node_rank):
        tmp_config = config.crop_config()
        path = tmp_config.module_profile_path(node_rank)
        module = get_module_info(path, module_name)
        fwd_time = module.get('time', float('inf')) * 1000 # us
        forward_step_time = get_module_info(path, 'forward_step_time') * 1000 # us
        backward_step_time = get_module_info(path, 'backward_step_time') * 1000 # us
        return fwd_time, fwd_time / forward_step_time * backward_step_time

    @staticmethod
    def get_chunks_time_module_level(config: ParallelConfig):
        forward_time_each_chunk = []
        backward_time_each_chunk = []
        nnodes = get_system_config().nnodes
        num_chunks = config.pipeline_model_parallel_size * config.virtual_pipeline_model_parallel_size
        for chunk_id in range(num_chunks):
            fwd_time = 0
            bwd_time = 0
            if chunk_id == 0:
                # chunk[0]包括embedding、transformer_layer
                embedding = TimeCostModel.get_module_time(config, 'embedding', 0)
                fwd_time += embedding[0]
                bwd_time += embedding[1]
                transfromer = TimeCostModel.get_module_time(config, '0', 0)
                fwd_time += transfromer[0] * config.num_layers_per_virtual_pipeline_stage
                bwd_time += transfromer[1] * config.num_layers_per_virtual_pipeline_stage
            elif chunk_id == num_chunks - 1:
                # chunk[-1]包括transformer_layer、final_layernorm、output_layer、loss
                transformer = TimeCostModel.get_module_time(config, '0', 0)
                fwd_time += transformer[0] * config.num_layers_per_virtual_pipeline_stage
                bwd_time += transformer[1] * config.num_layers_per_virtual_pipeline_stage
                final_norm = TimeCostModel.get_module_time(config, 'final_layernorm', 0)
                fwd_time += final_norm[0]
                bwd_time += final_norm[1]
                output_layer = TimeCostModel.get_module_time(config, 'output_layer', 0)
                fwd_time += output_layer[0]
                bwd_time += output_layer[1]
                loss = TimeCostModel.get_module_time(config, 'loss', 0)
                fwd_time += loss[0]
                bwd_time += loss[1]
            else:
                # chunk[mid]仅包括transformer layer
                transformer = TimeCostModel.get_module_time(config, '0', 0)
                fwd_time += transformer[0] * config.num_layers_per_virtual_pipeline_stage
                bwd_time += transformer[1] * config.num_layers_per_virtual_pipeline_stage

            forward_time_each_chunk.append(fwd_time)
            backward_time_each_chunk.append(bwd_time)

        return forward_time_each_chunk, backward_time_each_chunk
    
    @staticmethod
    def get_chunks_time_operator_level(config: ParallelConfig):
        forward_time_each_chunk = []
        backward_time_each_chunk = []
        num_chunks = config.pipeline_model_parallel_size * config.virtual_pipeline_model_parallel_size

        tf_block = TransformerBlock(config, TimeCostModel.number_samples)
        tf_block_fwd, tf_block_bwd = tf_block.get_block_time()

        for _ in range(num_chunks):
            forward_time_each_chunk.append(tf_block_fwd * config.num_layers_per_virtual_pipeline_stage)
            backward_time_each_chunk.append(tf_block_bwd * config.num_layers_per_virtual_pipeline_stage)

        return forward_time_each_chunk, backward_time_each_chunk
    
    @staticmethod
    def get_iteration_time(config: ParallelConfig, method: Literal['operator_level', 'module_level']):
        iteration_times = np.array([0 for _ in range(TimeCostModel.number_samples)]).astype(np.float64)

        if method == 'module_level':
            time_chunks = TimeCostModel.get_chunks_time_module_level(config)
            for i in range(TimeCostModel.number_samples):
                iteration_times[i], _ = TimeCostModel.pipeline_costmodel(config, time_chunks[0], time_chunks[1])
        else:
            time_chunks = TimeCostModel.get_chunks_time_operator_level(config)
            for i in range(TimeCostModel.number_samples):
                fwd_chunks = [chunk[i] for chunk in time_chunks[0]]
                bwd_chunks = [chunk[i] for chunk in time_chunks[1]]
                iteration_times[i], _ = TimeCostModel.pipeline_costmodel(config, fwd_chunks, bwd_chunks)
        
        return iteration_times
    
    @staticmethod
    def pipeline_costmodel(config: ParallelConfig, fwd_time_chunks, bwd_time_chunks):
        args = get_model_config().args

        send_recv_time = CommProfiling.get_send_recv_time(
            [args.seq_length, config.micro_batch_size, args.hidden_size]
        )

        comm_matrix = [[send_recv_time] * config.pipeline_model_parallel_size \
                       for _ in range(config.pipeline_model_parallel_size)]
        for i in range(config.pipeline_model_parallel_size):
            comm_matrix[i][i] = 0

        paras = PipelineParallelParas(
            num_stages=config.pipeline_model_parallel_size,
            vpp=config.virtual_pipeline_model_parallel_size,
            fwd_durations=fwd_time_chunks,
            bwd_durations=bwd_time_chunks,
            num_microbatches=config.num_microbatch,
            comm_matrix=comm_matrix
        )

        scheduler_1f1b = get_schedule_1f1b(paras)

        e2e_time_1f1b, stage_start_time = time_model_nfmb(paras, scheduler_1f1b)

        return e2e_time_1f1b, stage_start_time