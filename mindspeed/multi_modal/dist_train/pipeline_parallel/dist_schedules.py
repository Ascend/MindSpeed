# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import contextlib
from typing import Iterator, List, Union, Optional
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.enums import ModelType
import megatron.core.pipeline_parallel.schedules as schedules
from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_context_parallel_world_size,
    is_pipeline_stage_before_split,
    is_pipeline_stage_after_split,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank
)
from ..communication.dist_ranks_match import get_dst_ranks
from ..communication.dist_communication import generate_send_recv_ops, send_recv_tensor_list, send_recv
from ..config.dist_train_config import (
    get_dist_model_config,
    get_all_config_size,
    is_forward_only_model,
    is_use_send_recv_list
)


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        if get_args().dist_train:
            return forward_backward_pipelining_without_interleaving
        return get_forward_backward_func(*args, **kwargs)

    return wrapper


def p2p_ops_wrapper(p2p_ops):
    @wraps(p2p_ops)
    def wrapper(*args, **kwargs):
        arguments = get_args()
        if arguments.dist_train:
            return _p2p_ops(*args, **kwargs)
        return p2p_ops(*args, **kwargs)
    return wrapper


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    reqs = []
    # To prevent deadlocks caused by different pipeline stages receiving tensor simultaneously.
    if get_pipeline_model_parallel_rank(is_global=True) % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)
    return reqs


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    seq_length = seq_length // get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if is_pipeline_stage_before_split(rank):
            if is_use_send_recv_list():
                tensor_shapes = [
                    {'shape': (seq_length, micro_batch_size, config.hidden_size), 'dtype': config.params_dtype},
                ]
            else:
                tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            if is_use_send_recv_list():
                tensor_shapes = [
                    {'shape': ((decoder_seq_length, micro_batch_size, config.hidden_size)), 'dtype': config.params_dtype},
                    {'shape': ((seq_length, micro_batch_size, config.hidden_size)), 'dtype': config.params_dtype}
                ]
            else:
                tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
                tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        if is_use_send_recv_list():
            tensor_shapes = [
                {'shape': ((seq_length, micro_batch_size, config.hidden_size)), 'dtype': config.params_dtype},
            ]
        else:
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))

    return tensor_shapes


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """
    Run non-interleaved 1F1B schedule, with communication between pipeline stages.
    Returns dictionary with losses if the last stage, empty dict otherwise.
    """
    model_config = get_dist_model_config()
    if hasattr(model_config, 'forward_only'):
        forward_only = model_config.forward_only
    if isinstance(model, list):
        if len(model) != 1:
            raise RuntimeError("non-interleaved pipeline parallelism does not support model chunking")
        model = model[0]
    if isinstance(data_iterator, list):
        if len(data_iterator) != 1:
            raise RuntimeError("non-pipeline-parallel schedule does not support model chunking")
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    args = get_args()
    pp_rank = get_pipeline_model_parallel_rank()
    model_config = get_dist_model_config(rank=torch.distributed.get_rank())
    num_warmup_microbatches = 0
    for index in range(model_config.model_index, get_all_config_size()):
        num_warmup_microbatches += get_dist_model_config(global_index=index).pipeline_model_parallel_size
    num_warmup_microbatches = num_warmup_microbatches - pp_rank - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # For more details, please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    get_shape_func = schedules.get_tensor_shapes if not is_forward_only_model() else get_tensor_shapes

    recv_tensor_shapes = get_shape_func(
        rank=pp_rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    send_tensor_shapes = get_shape_func(
        rank=pp_rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )

    send_recv_ops = generate_send_recv_ops(torch.distributed.get_rank())

    if args.fp32_residual_connection:
        config.pipeline_dtype = torch.float32

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config, send_recv_ops)
        output_tensor = schedules.forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            schedules.check_first_val_step(first_val_step, forward_only, i == 0),
        )
        send_forward(output_tensor, send_tensor_shapes, config, send_recv_ops)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            schedules.deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config, send_recv_ops)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None
        output_tensor = schedules.forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            schedules.check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
        )

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config, send_recv_ops)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config, send_recv_ops)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config, send_recv_ops
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            schedules.deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or pp_rank == 0:
                    enable_grad_sync()
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config, send_recv_ops)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config, send_recv_ops
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or pp_rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config, send_recv_ops)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config, send_recv_ops)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func([model])

    return forward_data_store


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    if not is_use_send_recv_list():
        return schedules.backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
    
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None and x.requires_grad:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    output_tensors = []
    output_grad_tensors = []
    if output_tensor_grad[0] is None:
        # The last stage have no input gradients and only one loss is used to backward
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
    else:
        for output, grad in zip(output_tensor, output_tensor_grad):
            if output.requires_grad:
                output_tensors.append(output)
                output_grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors=output_grad_tensors)

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                if x.grad is None:
                    input_tensor_grad.append(torch.zeros_like(x, device=torch.cuda.current_device()))
                else:
                    input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        get_pipeline_model_parallel_world_size() > 1
        and is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def get_send_recv_fun():
    if is_use_send_recv_list():
        return send_recv_tensor_list
    else:
        return send_recv


def post_process_for_recving(recv_tensors: List):
    if is_use_send_recv_list():
        return [tensors[0] for tensors in recv_tensors]
    else:
        return [recv_tensors[0]]


def send_forward(output_tensors, tensor_shapes, config, send_recv_ops):
    if send_recv_ops['send_forward']:
        send_recv_func = get_send_recv_fun()
        send_recv_func(output_tensors, False, get_dst_ranks())
    else:
        schedules.send_forward(output_tensors, tensor_shapes, config)


def recv_forward(tensor_shapes, config, send_recv_ops):
    if send_recv_ops['recv_forward']:
        send_recv_func = get_send_recv_fun()
        recv_tensors = send_recv_func(None, True, get_dst_ranks())
        input_tensor = post_process_for_recving(recv_tensors)
    else:
        input_tensor = schedules.recv_forward(tensor_shapes, config)
    return input_tensor


def send_backward(input_tensor_grad, tensor_shapes, config, send_recv_ops):
    if send_recv_ops['send_backward']:
        send_recv_func = get_send_recv_fun()
        send_recv_func(input_tensor_grad, False, get_dst_ranks())
    else:
        schedules.send_backward(input_tensor_grad, tensor_shapes, config)


def send_forward_recv_backward(output_tensor, tensor_shape, config, send_recv_ops):
    if send_recv_ops['send_forward'] and send_recv_ops['recv_backward']:
        send_recv_func = get_send_recv_fun()
        recv_tensors = send_recv_func(output_tensor, True, get_dst_ranks())
        output_tensor_grad = post_process_for_recving(recv_tensors)
    else:
        output_tensor_grad = schedules.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, tensor_shapes, config, send_recv_ops):
    if send_recv_ops['recv_forward']:
        if not send_recv_ops['send_backward']:  # forward_only model adaption
            input_tensor_grad = [[]]
        send_recv_func = get_send_recv_fun()
        recv_tensors = send_recv_func(input_tensor_grad, True, get_dst_ranks())
        input_tensor = post_process_for_recving(recv_tensors)
    else:
        input_tensor = schedules.send_backward_recv_forward(
            input_tensor_grad, tensor_shapes, config
        )
    return input_tensor


def recv_backward(tensor_shapes, config, send_recv_ops):
    if send_recv_ops['recv_backward']:
        send_recv_func = get_send_recv_fun()
        recv_tensors = send_recv_func(None, True, get_dst_ranks())
        output_tensor_grad = post_process_for_recving(recv_tensors)
    else:
        output_tensor_grad = schedules.recv_backward(tensor_shapes, config)
    return output_tensor_grad
