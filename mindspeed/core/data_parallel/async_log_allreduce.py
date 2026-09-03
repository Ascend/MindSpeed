# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch
import megatron.training.training as megatron_training

from megatron.core import mpu
from mindspeed.args_utils import get_full_args as get_args
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_timers
from megatron.core.optimizer.qk_clip import clip_qk
from megatron.core.utils import unwrap_model
from megatron.training.utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine


def _validate_loss_keys(losses_reduced, loss_keys):
    """Reject inconsistent metric dictionaries before starting collectives."""
    expected_keys = set(loss_keys)
    for loss_dict in losses_reduced:
        if set(loss_dict) != expected_keys:
            raise ValueError(
                "All microbatches must report the same loss keys when using "
                f"--async-log-allreduce; expected {sorted(expected_keys)}, "
                f"but got {sorted(loss_dict)}."
            )


def start_async_loss_reductions(losses_reduced):
    """Start Megatron 0.18 loss reductions before the optimizer step.

    Megatron 0.18 stores one metric dictionary per microbatch. Two-element
    tensors contain a numerator and token count and still require a DP+CP
    reduction. Scalar tensors have already been reduced according to
    Megatron's native reporting contract.

    Returns a dictionary of ``(value, Work, is_ratio)`` records. MindSpeed
    owns every Work in this dictionary; ``loss_func`` must only return native
    metric dictionaries.
    """
    if not losses_reduced:
        return {}

    if not all(isinstance(record, dict) for record in losses_reduced):
        raise TypeError("Megatron 0.18 async loss records must be metric dictionaries.")

    loss_keys = list(losses_reduced[0])
    _validate_loss_keys(losses_reduced, loss_keys)
    reductions = {}
    for key in loss_keys:
        values = [record[key].view(-1) for record in losses_reduced]
        value_size = values[0].numel()
        if any(value.numel() != value_size for value in values):
            raise ValueError(f"Inconsistent value shapes across microbatches for loss key {key}.")

        if value_size == 2:
            value = torch.vstack(values).sum(dim=0)
            handle = torch.distributed.all_reduce(
                value,
                group=mpu.get_data_parallel_group(with_context_parallel=True),
                async_op=True,
            )
            reductions[key] = (value, handle, True)
        elif value_size == 1:
            reductions[key] = (torch.cat(values).mean(), None, False)
        else:
            raise ValueError(f"Invalid value shape: {values[0].shape} for key {key}")

    return reductions


def finish_async_loss_reductions(pending_reductions):
    """Wait for pending loss collectives and produce the logging dictionary."""
    loss_reduced = {}
    for key, (value, handle, is_ratio) in pending_reductions.items():
        if handle is not None:
            if not isinstance(handle, torch.distributed.Work):
                raise AssertionError(f"Expected {torch.distributed.Work} for loss key {key}, but got {type(handle)}.")
            handle.wait()
        loss_reduced[key] = value[0] / value[1] if is_ratio else value
    return loss_reduced


def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    forward_backward_func,
    iteration=None,
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    rerun_state_machine = get_rerun_state_machine()
    save_params_in_this_iteration = (
        args.save_params_interval is not None and (iteration + 1) % args.save_params_interval == 0
    )
    save_activations_in_this_iteration = (
        args.save_activations_interval is not None and (iteration + 1) % args.save_activations_interval == 0
    )
    save_tpe_in_this_iteration = (
        args.save_tokens_per_expert_interval is not None and (iteration + 1) % args.save_tokens_per_expert_interval == 0
    )
    save_wgrads_in_this_iteration = (
        args.save_wgrads_interval is not None and (iteration + 1) % args.save_wgrads_interval == 0
    )
    save_dgrads_in_this_iteration = (
        args.save_dgrads_interval is not None and (iteration + 1) % args.save_dgrads_interval == 0
    )
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
            model_chunk.force_all_reduce = save_wgrads_in_this_iteration
        optimizer.zero_grad()

        if megatron_training.has_nvidia_modelopt:
            adjust_tensor_shapes_fn = megatron_training.get_tensor_shapes_adjust_fn_for_distillation(
                model,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
            )
        else:
            adjust_tensor_shapes_fn = None

        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            forward_pre_hook_enabled = len(model[0].remove_forward_pre_hook_handles) > 0
            full_cg_captured = megatron_training.FullCudaGraphWrapper.cuda_graph.get("training") is not None
            if forward_pre_hook_enabled or full_cg_captured:
                for optim_instance in optimizer.chained_optimizers:
                    if isinstance(optim_instance, megatron_training.DistributedOptimizer):
                        optim_instance._copy_main_params_to_param_buffer()

        if save_activations_in_this_iteration:
            megatron_training.enable_activation_logging(model, args.save)
        if save_tpe_in_this_iteration:
            megatron_training.enable_tokens_per_expert_logging(model, args.save)
        if save_dgrads_in_this_iteration:
            megatron_training.enable_dgrad_logging(model, args.save)

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
            force_all_reduce=save_wgrads_in_this_iteration,
        )

        if save_activations_in_this_iteration:
            megatron_training.save_activations(iteration + 1)
            megatron_training.disable_activation_logging()
        if save_tpe_in_this_iteration:
            megatron_training.save_tokens_per_expert(iteration + 1)
            megatron_training.disable_tokens_per_expert_logging()
        if save_dgrads_in_this_iteration:
            megatron_training.save_dgrads(iteration + 1)
            megatron_training.disable_dgrad_logging()

        for model_chunk in model:
            model_chunk.force_all_reduce = False

    def _save_state_dict(attr_name, label):
        state_dict = defaultdict(dict)
        for model_chunk_id, model_chunk in enumerate(model):
            model_chunk_name = f"model_chunk{model_chunk_id}"
            unwrapped_model_chunk = unwrap_model(model_chunk)
            for param_name, param in unwrapped_model_chunk.named_parameters():
                if getattr(param, attr_name, None) is not None:
                    state_dict[model_chunk_name][param_name] = getattr(param, attr_name).cpu()

        megatron_training.save_grads(args.save, state_dict, iteration + 1, label)

    if save_wgrads_in_this_iteration:
        _save_state_dict(attr_name="main_grad", label="wgrads")

    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None, 0

    pending_loss_reductions = None
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        pending_loss_reductions = start_async_loss_reductions(losses_reduced)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, "vision_pretraining", False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    max_attention_logit = 0
    if getattr(args, "qk_clip", False) or getattr(args, "log_max_attention_logit", False):
        max_attention_logit = clip_qk(model, log_max_only=not getattr(args, "qk_clip", False))
    timers("optimizer").stop()

    if save_params_in_this_iteration:
        _save_state_dict(attr_name="data", label="params")

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if getattr(args, "vision_pretraining", False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = finish_async_loss_reductions(pending_loss_reductions)
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
            max_attention_logit,
        )
    return (
        {},
        skipped_iter,
        should_checkpoint,
        should_exit,
        exit_code,
        grad_norm,
        num_zeros_in_grad,
        max_attention_logit,
    )
