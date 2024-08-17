# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
"""Pretrain utilities."""

import gc
from functools import wraps
import torch
import torch_npu
from megatron.training import get_args
from megatron.training import get_timers
from megatron.core import parallel_state
from megatron.training import print_rank_0
from megatron.training.arguments import parse_args
from mindspeed.core.memory.auto_pipeline.autopipeline import autopipeline_profiling
from mindspeed.core.performance.auto_pipeline_perf.autopipeline_perf import (autopipelineperf_profiling, check_out_of_memory,
                                                                             calculate_num_of_activations, check_skip_profiling,
                                                                             broadcast_skip_in_ranks)
from mindspeed.core.performance.auto_pipeline_perf.optimpipeline_solver import solve_optimpipeline, broadcast_oom_in_ranks, broadcast_mbs_in_ranks, save_profiling_data
from mindspeed.core.performance.auto_pipeline_perf.schedulepipeline_solver import (solve_pipelineschedule, broadcast_enable_schedule_in_ranks,
                                                                                   broadcast_scheduler_in_ranks, broadcast_layer_in_ranks,
                                                                                   all_gather_time, average_time_by_rank)
from mindspeed.core.memory.auto_pipeline.autopipeline_apply import apply_autopipeline
from mindspeed.core.memory.auto_pipeline.autopipeline_solver import solve_autopipeline, broadcast_policy_in_ranks, destroy_global_vars
from mindspeed.arguments import parse_args_wrapper


POLICY = None
OPTIMIZED_MBS_LIST = None
PP_SCHEDULE_LIST = None
OPTIMAL_LAYERS = None
ORIGIN_MBS = None
DATA_PARALLEL_SIZE = 1
ENABLE_SCHEDULER = False


def train_decorator(train):
    @wraps(train)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        if args_.profile_npu and (torch.distributed.get_rank() in args_.profile_ranks):
            active = args_.profile_step_end - args_.profile_step_start
            skip_first = args_.profile_step_start

            if args_.profile_with_cpu:
                activities = [torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU]
            else:
                activities = [torch_npu.profiler.ProfilerActivity.NPU]

            if args_.profile_level == 'level0':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level0
            elif args_.profile_level == 'level1':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level1
            elif args_.profile_level == 'level2':
                profiler_level = torch_npu.profiler.ProfilerLevel.Level2
            else:
                raise ValueError(f"profiler_level only support level0, level1, level2, but gets {args_.profile_level}")

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=profiler_level,
                l2_cache=False
            )

            with torch_npu.profiler.profile(
                activities=activities,
                record_shapes=args_.profile_record_shapes,
                profile_memory=args_.profile_with_memory,
                with_stack=args_.profile_with_stack,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args_.profile_save_path)
            ) as prof:
                args_.prof = prof
                return train(*args, **kwargs)
        else:
            return train(*args, **kwargs)

    return wrapper


def train_step_decorator(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        ret = train_step(*args, **kwargs)
        args_ = get_args()
        if args_.profile_npu and (torch.distributed.get_rank() in args_.profile_ranks):
            args_.prof.step()
        return ret
    return wrapper


def pretrain_decorator(pretrain):
    @wraps(pretrain)
    def wrapper(*args, **kwargs):
        global POLICY
        global OPTIMIZED_MBS_LIST
        global PP_SCHEDULE_LIST
        global OPTIMAL_LAYERS
        global ORIGIN_MBS
        global DATA_PARALLEL_SIZE
        global ENABLE_SCHEDULER
        new_parse_args = parse_args_wrapper(parse_args)
        argument = new_parse_args(kwargs.get('extra_args_provider'), False)
        if argument.automated_pipeline and not argument.num_layer_list:
            context, POLICY = autopipeline_profiling(args[1], args[2], args[3],
                                                     args[0], None, argument)
            if context:
                POLICY = solve_autopipeline(context)
                parallel_state.destroy_global_memory_buffer()
                parallel_state.destroy_model_parallel()
                destroy_global_vars()
                gc.collect()
                torch.cuda.empty_cache()

        if argument.automated_pipeline_perf:
            ORIGIN_MBS = argument.micro_batch_size
            is_skip, exist_policy = check_skip_profiling(argument, config_file="autopipeline_perf_config.json")
            if not is_skip:
                global_context = []
                mbs_time, pp_schedule_time = 0, 0
                mbs_tries = 1
                num_forwards_first_stage = 0
                is_oom = False
                forward_time_dict = {}
                backward_time_dict = {}

                while mbs_tries < ORIGIN_MBS + 2:
                    context = autopipelineperf_profiling(mbs_tries, args[1], args[2], args[3],
                                                              args[0], None)
                    if mbs_tries == ORIGIN_MBS:
                        schedule_context = context
                        forward_time_list = all_gather_time(argument, schedule_context['fwd_time'])
                        forward_time_dict = average_time_by_rank(forward_time_list)
                        backward_time_list = all_gather_time(argument, schedule_context['bwd_time'])
                        backward_time_dict = average_time_by_rank(backward_time_list)
                        num_forwards_first_stage = calculate_num_of_activations(schedule_context)

                    parallel_state.destroy_global_memory_buffer()
                    parallel_state.destroy_model_parallel()
                    destroy_global_vars()
                    gc.collect()
                    torch.cuda.empty_cache()
                    global_context.append((context['fwd_time'], context['bwd_time'], context['comm_time']))
                    DATA_PARALLEL_SIZE = context['data_parallel_size']
                    if not is_oom:
                        is_oom = check_out_of_memory(argument, context, mbs_tries)
                        is_oom = broadcast_oom_in_ranks(0, is_oom)
                    mbs_tries += 1
                    if mbs_tries <= ORIGIN_MBS and is_oom:
                        raise AssertionError(
                        'A risk of Out of Memory could occur, please '
                        'reset to a smaller micro batch size.')
                    if mbs_tries > ORIGIN_MBS and is_oom:
                        break
                if len(global_context) > 0:
                    OPTIMIZED_MBS_LIST, mbs_time = solve_optimpipeline(argument, DATA_PARALLEL_SIZE, global_context)
                PP_SCHEDULE_LIST, pp_schedule_time, OPTIMAL_LAYERS = solve_pipelineschedule(argument, DATA_PARALLEL_SIZE, num_forwards_first_stage, forward_time_dict, backward_time_dict)
                if torch.distributed.get_rank() == 0 and mbs_time > pp_schedule_time and num_forwards_first_stage > 2:
                    ENABLE_SCHEDULER = True
                ENABLE_SCHEDULER = broadcast_enable_schedule_in_ranks(0, ENABLE_SCHEDULER)
                optimized_policy = (ENABLE_SCHEDULER, OPTIMIZED_MBS_LIST, PP_SCHEDULE_LIST, OPTIMAL_LAYERS)
                save_profiling_data(optimized_policy, config_file="autopipeline_perf_config.json")
            else:
                ENABLE_SCHEDULER = exist_policy[0]
                OPTIMIZED_MBS_LIST = exist_policy[1]
                PP_SCHEDULE_LIST = exist_policy[2]
                OPTIMAL_LAYERS = exist_policy[3]
        pretrain(*args, **kwargs)
    return wrapper


def setup_model_and_optimizer_decorator(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(*args, **kwargs):
        global POLICY
        global OPTIMIZED_MBS_LIST
        global PP_SCHEDULE_LIST
        global OPTIMAL_LAYERS
        global ENABLE_SCHEDULER
        argument = get_args()
        if argument.automated_pipeline and POLICY:
            if torch.distributed.get_rank() == 0:
                broadcast_policy_in_ranks(0, policy)
            else:
                broadcast_policy_in_ranks(0)
        if argument.automated_pipeline_perf and ENABLE_SCHEDULER:
            broadcast_scheduler_in_ranks(0, PP_SCHEDULE_LIST)
            broadcast_layer_in_ranks(0, OPTIMAL_LAYERS)
        elif argument.automated_pipeline_perf and OPTIMIZED_MBS_LIST:
            broadcast_mbs_in_ranks(0, OPTIMIZED_MBS_LIST)
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(*args, **kwargs)
        if argument.recompute_module_list:
            apply_autopipeline(model)
        return model, optimizer, opt_param_scheduler
    return wrapper


def save_checkpoint_and_time_decorator(save_checkpoint_and_time):
    @wraps(save_checkpoint_and_time)
    def wrapper(*args, **kwargs):
        global_args = get_args()
        optimizer = args[2]
        if global_args.use_distributed_optimizer and global_args.overlap_param_gather:
            optimizer.disable_pre_hook()
        save_checkpoint_and_time(*args, **kwargs)
        if global_args.use_distributed_optimizer and global_args.overlap_param_gather:
            optimizer.enable_pre_hook()
    return wrapper
