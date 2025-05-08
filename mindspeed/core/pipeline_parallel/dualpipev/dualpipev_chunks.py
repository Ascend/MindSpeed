# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import time
from functools import wraps
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.training.global_vars import get_args, get_timers
from megatron.training.utils import unwrap_model, print_rank_0, is_last_rank
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.legacy.model.module import fp32_to_float16, float16_to_fp32
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state
from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import get_dualpipe_chunk
from mindspeed.core.data_parallel.async_log_allreduce import get_async_reduced_loss_value


def dualpipev_fp16forward(self, *inputs, **kwargs):
    dualpipe_first_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 0
    if dualpipe_first_stage:
        inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    dualpipe_last_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
    if dualpipe_last_stage:
        outputs = float16_to_fp32(outputs)
    return outputs


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    assert model_type != ModelType.encoder_and_decoder, \
        "Interleaved schedule not supported for model with both encoder and decoder"
    model = []

    pre_process, post_process = False, False
    if mpu.is_pipeline_first_stage():
        pre_process = True

    args.dualpipev_first_chunk = True
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
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
                  mpu.get_tensor_model_parallel_rank(),
                  mpu.get_pipeline_model_parallel_rank(),
                  sum([sum([p.nelement() for p in model_module.parameters()])
                       for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=args.ddp_average_in_collective)
        model = [DDP(config,
                     ddp_config,
                     model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
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

        if args.async_log_allreduce:
            # when async_log_allreduce is on, loss_reduced is list[tuple(dict,torch.distributed.group)]
            losses_reduced_keys = losses_reduced[0][0].keys()
        else:
            losses_reduced_keys = losses_reduced[0].keys()

        for key in losses_reduced_keys:
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                if args.async_log_allreduce:
                    val = get_async_reduced_loss_value(x, key)
                else:
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
            # embedding and lm head are on save rank.
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
             verbose=False):
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
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
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
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
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

    return total_loss_dict, collected_non_loss_data, False
