# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import time
import dataclasses
from functools import wraps
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
from megatron.core.enums import ModelType
from megatron.training import ft_integration, one_logger_utils
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training.global_vars import get_args, get_timers, get_one_logger, get_wandb_writer
from megatron.training.utils import unwrap_model, print_rank_0, is_last_rank, append_to_progress_log
from megatron.training.training import print_datetime, setup_model_and_optimizer, build_train_valid_test_data_iterators, train, preprocess_common_state_dict, evaluate_and_print_results
from megatron.training.checkpointing import save_checkpoint
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.legacy.model.module import fp32_to_float16, float16_to_fp32
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.rerun_state_machine import RerunMode

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import get_dualpipe_chunk
from mindspeed.core.pipeline_parallel.dualpipev.mtp_utils import model_provider_mtp


def get_transformer_layer_offset(config: TransformerConfig):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    args = get_args()
    num_layers_per_pipeline_rank = config.num_layers // config.pipeline_model_parallel_size
    num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2
    if getattr(args, 'dualpipev_first_chunk', True):
        offset = pipeline_rank * num_layers_per_pipeline_rank
    else:
        offset = config.num_layers - (pipeline_rank + 1) * num_layers_per_pipeline_rank
    return offset


def dualpipev_fp16forward(self, *inputs, **kwargs):
    dualpipe_first_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 0
    if dualpipe_first_stage:
        inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    dualpipe_last_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
    if dualpipe_last_stage:
        outputs = float16_to_fp32(outputs)
    return outputs


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Initialize fault tolerance
    # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
    if args.enable_ft_package:
        ft_integration.setup(args)
        ft_integration.maybe_setup_simulated_fault()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    from megatron.training.training import _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
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

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                LocalCheckpointManager
            from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import \
                parse_group_sequence, GroupWrapper
            from nvidia_resiliency_ext.checkpointing.local.replication.strategies import \
                CliqueReplicationStrategy
        except ModuleNotFoundError:
            raise RuntimeError("The 'nvidia_resiliency_ext' module is required for local "
                               "checkpointing but was not found. Please ensure it is installed.")

        if args.replication:
            repl_strategy = CliqueReplicationStrategy.from_replication_params(
                args.replication_jump,
                args.replication_factor
            )
        else:
            repl_strategy = None

        checkpointing_context = {
            'local_checkpoint_manager': LocalCheckpointManager(args.non_persistent_local_ckpt_dir,
                                                               repl_strategy=repl_strategy
                                                               )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()

    #If with MTP, change model_provider func.
    if args.mtp_num_layers is not None:
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider_mtp, model_type, checkpointing_context=checkpointing_context)
    else:
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type, checkpointing_context=checkpointing_context)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()
    config = get_model_config(model[0])

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
    elif args.schedules_method == 'dualpipev':
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for _ in range(2):
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
                process_non_loss_data_func, config, checkpointing_context,
                non_loss_data_func)

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context,
                            train_data_iterator=train_data_iterator,
                            preprocess_common_state_dict_fn=preprocess_common_state_dict)

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
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })

    ft_integration.shutdown()
    one_logger_utils.finish()


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    use_dualpipe_mtp = None
    if args.mtp_num_layers is not None:
        use_dualpipe_mtp = False

    assert model_type != ModelType.encoder_and_decoder, \
        "Interleaved schedule not supported for model with both encoder and decoder"
    model = []

    pre_process, post_process = False, False
    if mpu.is_pipeline_first_stage():
        pre_process = True
        use_dualpipe_mtp = True
        
    args.dualpipev_first_chunk = True
    if args.mtp_num_layers:
        first_model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process,
            use_dualpipe_mtp=False
        )
        first_model.model_type = model_type
        model.append(first_model)

        args.dualpipev_first_chunk = False
        second_model = model_provider_func(
            pre_process=post_process,
            post_process=pre_process,
            use_dualpipe_mtp=use_dualpipe_mtp
        )
        second_model.model_type = model_type
        model.append(second_model)
    else:
        first_model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process
        )
        first_model.model_type = model_type
        model.append(first_model)

        args.dualpipev_first_chunk = False
        second_model = model_provider_func(
            pre_process=post_process,
            post_process=pre_process
        )
        second_model.model_type = model_type
        model.append(second_model)

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    num_parameters = sum(
        [sum([p.nelement() for p in model_module.parameters()])
         for model_module in model]
    )
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            num_parameters), flush=True)

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if not (args.use_torch_fsdp2 and args.use_cpu_initialization) and not args.init_model_with_meta_device:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif args.use_custom_fsdp:
            DP = custom_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
        kwargs['check_for_large_grads'] = args.check_for_large_grads
        if args.ddp_num_buckets is not None:
            assert args.ddp_bucket_size is None, \
                "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
            assert args.ddp_num_buckets > 0, \
                "--ddp-num-buckets must be greater than 0"
            kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
        else:
            kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        if args.use_custom_fsdp and args.use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False
        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not getattr(args, "use_torch_fsdp2", False):
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [DP(config=config,
                     ddp_config=ddp_config,
                     module=model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step)
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config): 
    """Single training step."""
    args = get_args()
    timers = get_timers()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
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
 
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

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

    dualpipevlaststage = mpu.is_pipeline_first_stage(ignore_virtual=True)
    if dualpipevlaststage:
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
        
        return loss_reduced, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad


def get_num_layers_to_build(config: TransformerConfig) -> int:

    num_layers_per_pipeline_rank = (
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )

    num_layers_to_build = num_layers_per_pipeline_rank // 2

    return num_layers_to_build


def _allreduce_embedding_grads_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().schedules_method == 'dualpipev':
            # dualpipev no need to do embedding allreduce
            # embedding and lm head are on same rank.
            if not get_args().untie_embeddings_and_output_weights:
                raise NotImplementedError
            else:
                return
        else:
            return fn(*args, **kwargs)

    return wrapper


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False,
             non_loss_data_func=None):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            ft_integration.on_eval_step_start()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
            ft_integration.on_eval_step_end()
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            # Change last stage to first stage for dualpipev
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if key not in total_loss_dict:
                            total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                        val = loss_dict[key]
                        if isinstance(val, tuple) or isinstance(val, list):
                            total_loss_dict[key][0] += val[0]
                            total_loss_dict[key][1] += val[1]
                        else:
                            total_loss_dict[key][0] += val
                            total_loss_dict[key][1] += 1

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                from megatron.training.training import _TRAIN_START_TIME
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers('evaluate').stop()
    timers.log(['evaluate'])

    rerun_state_machine.set_mode(rerun_mode)

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            'position_ids': data["position_ids"].cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            if args.schedules_method == "dualpipev":
                _broadcast(batch['loss_mask'])
                _broadcast(batch['labels'])

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if args.mtp_num_layers is not None:
                _broadcast(batch['tokens'])
                _broadcast(batch['position_ids'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                    (args.micro_batch_size,1,args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device()
                )
        else:
            attention_mask = None
        position_ids=torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            if args.schedules_method == "dualpipev":
                _broadcast(loss_mask)
                _broadcast(labels)
            else:
                labels = None
                loss_mask = None 

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            if args.mtp_num_layers is not None:
                _broadcast(tokens)
                _broadcast(position_ids)
            else:
                tokens = None
                position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch
