# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
"""Pretrain utilities."""

import os
import sys
import gc
import os
from functools import wraps
import torch
import torch_npu
from datetime import datetime
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import is_last_rank
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.training import print_rank_0
from megatron.training.arguments import parse_args
from megatron.training.global_vars import (set_args, get_tensorboard_writer, get_wandb_writer,
                                           get_one_logger)
from megatron.training.training import num_floating_point_operations
from megatron.training.utils import print_rank_last, report_memory
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from mindspeed.core.auto_parallel.auto_parallel_apply import search_optimal_configuration
from mindspeed.core.auto_parallel.auto_parallel_profiling import Profiling, OperateProfile
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
FLOPS_COUNTER = None
RECORDED_COUNT = 0
TRAVERSED_COUNT = 0


def generated_flops_counter():
    from torch_npu.utils.flops_count import FlopsCounter
    global FLOPS_COUNTER
    FLOPS_COUNTER = FlopsCounter()


def get_flops_counter():
    global FLOPS_COUNTER
    if FLOPS_COUNTER is None:
        generated_flops_counter()
    return FLOPS_COUNTER


def set_count(count):
    global RECORDED_COUNT
    global TRAVERSED_COUNT
    RECORDED_COUNT = count[0]
    TRAVERSED_COUNT = count[1]


def get_count():
    global RECORDED_COUNT
    global TRAVERSED_COUNT
    if RECORDED_COUNT == 0 and TRAVERSED_COUNT == 0:
        flops_counter = get_flops_counter()
        count = flops_counter.get_flops()
        set_count(count)
    return RECORDED_COUNT, TRAVERSED_COUNT


def train_decorator(train):
    @wraps(train)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        is_profile = hasattr(args_, 'profile_npu') and args_.profile_npu \
                and ((torch.distributed.get_rank() in args_.profile_ranks) or (-1 in args_.profile_ranks))
        if is_profile:
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
        nonlocal train_step
        args_ = get_args()
        flop_count = None
        if args_.op_cal_tflops:
            flop_count = get_flops_counter()
            flop_count.start()
        if args_.profile_operator:
            op_profile = OperateProfile(args_)
            ret = train_step(*args, **kwargs)
            op_profile.step()
        elif args_.prof_file:
            profiling = Profiling(args_)
            train_step = profiling.hook_train_step(train_step)
            ret = train_step(*args, **kwargs)
        else:
            ret = train_step(*args, **kwargs)
            is_profile = args_.profile_npu and ((torch.distributed.get_rank() in args_.profile_ranks) or (-1 in args_.profile_ranks))
            if is_profile:
                args_.prof.step()
        if args_.op_cal_tflops:
            counts = flop_count.get_flops()
            set_count(counts)
            flop_count.stop()
        return ret
    return wrapper


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    if one_logger:
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)

        # select all nodes info
        counts_0, counts_1 = get_count()
        counts_0_tensor = torch.tensor([counts_0], device="npu")
        counts_1_tensor = torch.tensor([counts_1], device="npu")

        torch.distributed.all_reduce(
            counts_0_tensor, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            counts_1_tensor, op=torch.distributed.ReduceOp.SUM
        )

        mfu = counts_0_tensor.cpu().item() / (10 ** 12 * elapsed_time_per_iteration * args.world_size)
        hfu = counts_1_tensor.cpu().item() / (10 ** 12 * elapsed_time_per_iteration * args.world_size)

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' theoretical throughput per NPU (TFLOP/s/NPU): {throughput:.1f} |'
            log_string += f' actual throughput per NPU (TFLOP/s/NPU): {mfu:.1f} |'
            log_string += f' actual throughput per NPU with recompute (TFLOP/s/NPU): {hfu:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (parallel_state.is_pipeline_first_stage(ignore_virtual=True) or
                                              parallel_state.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


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

        if argument.auto_tuning:
            set_args(argument)
            print("pretrain_decorator set_args ========================================")

            from mindspeed.auto_tuning.auto_tuning import auto_tuning
            global_args = get_args()
            assert global_args.auto_tuning_ranks >= 16, "Auto-tuning searching space should be >= 16."
            working_dir_root = os.path.realpath(global_args.auto_tuning_work_dir)
            if not os.path.exists(working_dir_root) and global_args.rank % torch.cuda.device_count() == 0:
                os.makedirs(working_dir_root)

            if global_args.rank % torch.cuda.device_count() == 0:
                print("only rank 0 run auto tuning ========================================")
                auto_tuning(global_args, working_dir=working_dir_root)
            return

        if argument.auto_parallel:
            set_args(argument)
            search_optimal_configuration(argument)
            return
        
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
                broadcast_policy_in_ranks(0, POLICY)
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
        if argument.profile_memory and torch.distributed.get_rank() in argument.profile_ranks:
            profiling = Profiling(argument)
            profiling.register_recursive_hook("", model)
        return model, optimizer, opt_param_scheduler
    return wrapper
