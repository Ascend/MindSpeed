# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import gc
import time
from functools import wraps
import torch

from megatron.core import mpu
from megatron.core.utils import get_model_config

from megatron.training.checkpointing import save_checkpoint

from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.initialize import write_args_to_tensorboard

from megatron.training.training import append_to_progress_log
from megatron.training.training import setup_model_and_optimizer
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.training import train
from megatron.training.training import evaluate_and_print_results
from megatron.training.training import print_datetime
from megatron.training.training import train_step
from megatron.training.training import training_log
from megatron.training.training import save_checkpoint_and_time
from megatron.training.training import num_floating_point_operations

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads

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
    get_one_logger,
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)
_TRAIN_START_TIME = time.time()


def train_uce(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    one_logger = get_one_logger()
    if one_logger:
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics({
            'train_samples_start': args.consumed_train_samples,
            'train_iterations_start': iteration,
            'train_samples_target': train_samples_target,
            'train_iterations_target': args.train_iters,
        })

    num_floating_point_operations_so_far = 0

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce and config.no_sync_func is None:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    exit = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def track_e2e_metrics():
        # Nested function to track a bunch of E2E APP metrics
        if one_logger:
            train_duration = timers('interval-time').active_time()  # overall_elapsed
            train_samples = args.consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (
                                                          train_duration * 1000.0) / train_iterations if train_iterations > 0 else None
            if eval_iterations > 0:
                validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            one_logger.log_metrics({
                'train_iterations_end': iteration,
                'train_samples_end': args.consumed_train_samples,
                'train_iterations': train_iterations,
                'train_samples': train_samples,
                'train_iterations_time_msecs_avg': train_iterations_time_msecs_avg,
                'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg
            })

    while iteration < args.train_iters:

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far)
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        iteration += 1
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
        num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)

        if args.enable_high_availability:
            from mindio_ttp.framework_ttp import tft_set_step_args

            args.num_floating_point_operations_so_far = num_floating_point_operations_so_far
            tft_set_step_args(
                [iteration, model, optimizer, opt_param_scheduler, args.num_floating_point_operations_so_far])
            args.iteration = iteration

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:
            track_e2e_metrics()

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          learning_rate,
                                          decoupled_learning_rate,
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)

        # Autoresume
        if args.adlr_autoresume and \
                (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
                args.do_valid:
            timers('interval-time').stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = 'iteration {}'.format(iteration)
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()
            timers('interval-time', log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \
                iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far)
            saved_checkpoint = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler,
                                             num_floating_point_operations_so_far)
                print_datetime('exiting program after {} minutes'.format(train_time))
                exit = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    track_e2e_metrics()

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if args.use_distributed_optimizer and args.overlap_param_gather:
        optimizer.disable_pre_hook()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:
        sys.exit()

    return iteration, num_floating_point_operations_so_far


def build_train_args(*input_args):
    args, timers, train_valid_test_dataset_provider, model_provider, model_type, forward_step_func, process_non_loss_data_func = input_args

    from megatron.training.training import setup_model_and_optimizer
    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
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

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    train_args = [forward_step_func,
                  model, optimizer, opt_param_scheduler,
                  train_data_iterator, valid_data_iterator, process_non_loss_data_func, config]
    test_data_iterator_list = [test_data_iterator]
    return train_args, test_data_iterator_list


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
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    train_args, test_data_iterator_list = build_train_args(args, timers, train_valid_test_dataset_provider,
                                                           model_provider, model_type, forward_step_func,
                                                           process_non_loss_data_func)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            if args.enable_high_availability:
                from mindio_ttp.adaptor import tft_init_controller_processor, tft_register_processor, tft_train

                tft_init_controller_processor(enable_tls=False, tls_option_top_path='')
                tft_register_processor(train_valid_test_dataset_provider, model_provider, model_type)
                iteration, num_floating_point_operations_so_far = tft_train(train_args, test_data_iterator_list)
            else:
                iteration, num_floating_point_operations_so_far = train(*train_args)
            test_data_iterator = test_data_iterator_list[0]
            forward_step_func, model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator, process_non_loss_data_func, config = train_args

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far)
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


def num_floating_point_wrapper(fn):
    @wraps(fn)
    def wrapper(args, batch_size):
        args.num_layers -= len(args.noop_layers) if isinstance(args.noop_layers, set) else 0
        res = fn(args, batch_size)
        args.num_layers += len(args.noop_layers) if isinstance(args.noop_layers, set) else 0
        return res

    return wrapper
