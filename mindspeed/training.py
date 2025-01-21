# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import time
from functools import wraps
import os
import torch

from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config

from megatron.training import one_logger_utils

from megatron.training.checkpointing import save_checkpoint

from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options

from megatron.training.training import append_to_progress_log
from megatron.training.training import setup_model_and_optimizer
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.training import train
from megatron.training.training import evaluate_and_print_results
from megatron.training.training import print_datetime
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)
from megatron.training.utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model)
from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger)

from mindspeed.auto_tuning.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling

_TRAIN_START_TIME = time.time()


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999
) -> None:
    """
    Step the EMA model towards the current model.
    """
    from collections import OrderedDict
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        param_data = param.data
        ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    if args.use_ema:
        unwrapped_model = unwrap_model(model)
        for model_chunk in unwrapped_model:
            update_ema(model_chunk.ema, model_chunk, optimizer=optimizer)


    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    if (os.getenv("OOTB_OPTIMIZER_PARSE_ARGS", "FALSE") == "TRUE"):
        working_dir = get_args().profile_save_path
        from mindspeed.auto_tuning.mindspeed_adaptor import MindSpeedAdaptor
        hardware = MindSpeedAdaptor.get_hardware(working_dir=working_dir)
        MindSpeedAdaptor.get_model_args(get_args(), hardware, working_dir)
        print_rank_0("================OOTB_OPTIMIZER_PARSE_ARGS END EXIT!====================")

        return

    if 'init_func' in args_defaults:
        init_func = args_defaults['init_func']
        init_func()

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.npu.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    args = get_args()
    timers = get_timers()

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()
    config = get_model_config(model[0])

    if (os.getenv("OOTB_OPTIMIZER_PARSE_MODEL", "FALSE") == "TRUE"):
        output_path = args.profile_save_path
        from mindspeed.auto_tuning.mindspeed_adaptor import MindSpeedAdaptor
        hardware = MindSpeedAdaptor.get_hardware()
        MindSpeedAdaptor.get_model_params(model, mpu.get_pipeline_model_parallel_rank(), hardware, output_path)
        print_rank_0("================OOTB_OPTIMIZER_PARSE_MODEL END EXIT!====================")
        return

    # Data stuff.
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(args.train_iters, args.skip_train, args.do_train,
                                        args.do_valid, args.do_test, args.dataloader_type,
                                        args.retro_project_dir, args.retro_cyclic_train_iters)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = {}

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config, checkpointing_context)

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context)

        one_logger and one_logger.log_metrics({
            'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()
        })

    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE':
        # profiling parser
        res_dir = args.profile_save_path
        cur_rank = torch.distributed.get_rank()
        if res_dir and cur_rank % 8 == 0:
            GatherNodeProfiling(res_dir).parse_node_pkl(args)
        print_datetime('after training is done')

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })
    one_logger_utils.finish()


def num_floating_point_wrapper(fn):
    @wraps(fn)
    def wrapper(args, batch_size):
        args.num_layers -= len(args.noop_layers) if isinstance(args.noop_layers, set) else 0
        res = fn(args, batch_size)
        args.num_layers += len(args.noop_layers) if isinstance(args.noop_layers, set) else 0
        return res

    return wrapper


def get_device_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = torch.distributed.get_backend()
        local_rank = args[0]
        if backend == 'hccl':
            if local_rank is None:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{local_rank}')
        else:
            device = func(*args, **kwargs)
        return device
    return wrapper
