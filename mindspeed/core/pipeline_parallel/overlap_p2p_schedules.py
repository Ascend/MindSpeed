# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
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

from typing import Iterator, List, Union
import torch
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import (
    forward_step,
    backward_step,
    deallocate_output_tensor,
    check_first_val_step
)
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
)
from megatron.training import get_args
from mindspeed.core.parallel_state import get_pipeline_parallel_group_for_new_stream
from mindspeed.core.pipeline_parallel.flexible_schedules import (
    send_forward,
    send_backward
)
from mindspeed.core.pipeline_parallel.interleaving import (
    InterleaveScheduleInfo,
    NanoPipe,
)

stream_ping = None
stream_pang = None
default_stream = None


def forward_backward_pipelining_with_interleaving_overlap_warmup_cooldown_p2p(
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
    Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise.
    """
    if not isinstance(model, list): 
        raise AssertionError("interleaved pipeline parallelism expected model chunking")
    if not all(isinstance(chunk, torch.nn.Module) for chunk in model): 
        raise AssertionError("invalid model chunking")
    if not isinstance(
        data_iterator, list
    ): 
        raise AssertionError("interleaved pipeline parallelism expected each model chunk to have a data iterator")
    args = get_args()
    config = get_model_config(model[0])
    if config.batch_p2p_comm:
        raise ValueError("Can not use batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def wait_helper(wait_handler):
        if wait_handler is not None:
            for req in wait_handler:
                req.wait()

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    InterleaveScheduleInfo.init_info(pipeline_parallel_size=pipeline_parallel_size,
                                     num_model_chunks=num_model_chunks,
                                     total_num_microbatches=total_num_microbatches,
                                     config=config)
    
    InterleaveScheduleInfo.disable_grad_sync()

    if args.use_nanopipe:
        NanoPipe.init_decouple(pipeline_parallel_size,
                               num_model_chunks,
                               total_num_microbatches,
                               num_warmup_microbatches,
                               args.use_nanopipe_swap)

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def forward_step_helper(microbatch_id, current_microbatch, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                param_sync_microbatch_id < total_num_microbatches
                and InterleaveScheduleInfo.is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        micro_batch_id = InterleaveScheduleInfo.get_microbatch_id_in_model_chunk(microbatch_id, forward=True)


        input_tensor = input_tensors[model_chunk_id][micro_batch_id]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, InterleaveScheduleInfo.is_first_microbatch_for_model_chunk(microbatch_id),
            ),
            current_microbatch=current_microbatch,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()
        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(microbatch_id, forward=False)
        micro_batch_data_id = InterleaveScheduleInfo.get_microbatch_id_in_model_chunk(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and InterleaveScheduleInfo.is_last_microbatch_for_model_chunk(microbatch_id):
            if not args.use_nanopipe or NanoPipe.nano_flag[model_chunk_id]:
                InterleaveScheduleInfo.enable_grad_sync()
                synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)

        input_tensor = input_tensors[model_chunk_id][micro_batch_data_id]
        output_tensor = output_tensors[model_chunk_id][micro_batch_data_id]
        output_tensor_grad = output_tensor_grads[model_chunk_id][micro_batch_data_id]

        # help free tensor memory
        input_tensors[model_chunk_id][micro_batch_data_id] = None
        output_tensors[model_chunk_id][micro_batch_data_id] = None
        output_tensor_grads[model_chunk_id][micro_batch_data_id] = None

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and InterleaveScheduleInfo.is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                if args.use_nanopipe:
                    NanoPipe.nanopipe_grad_sync(grad_sync_chunk_id, synchronized_model_chunks, config, model)
                else:
                    InterleaveScheduleInfo.enable_grad_sync()
                    config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                    synchronized_model_chunks.add(grad_sync_chunk_id)
        InterleaveScheduleInfo.disable_grad_sync()
        return input_tensor_grad

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    group_ping = get_pipeline_model_parallel_group()
    group_pang = get_pipeline_parallel_group_for_new_stream()

    # use ping pang buffer to parallel send and receive
    global default_stream
    if default_stream is None:
        default_stream = torch.cuda.default_stream()

    if args.use_multi_stream:
        global stream_ping
        if stream_ping is None:
            stream_ping = torch.cuda.Stream()

        global stream_pang
        if stream_pang is None:
            stream_pang = torch.cuda.Stream()


        if get_pipeline_model_parallel_rank() % 2 == 0:
            receive_stream = stream_ping
            send_stream = stream_pang
            receive_group = group_ping
            send_group = group_pang
        else:
            receive_stream = stream_pang
            send_stream = stream_ping
            receive_group = group_pang
            send_group = group_ping
    else:
        if get_pipeline_model_parallel_rank() % 2 == 0:
            receive_stream = default_stream
            send_stream = default_stream
            receive_group = group_ping
            send_group = group_pang
        else:
            receive_stream = default_stream
            send_stream = default_stream
            receive_group = group_pang
            send_group = group_ping       

    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    with torch.cuda.stream(receive_stream):
        input_tensor, wait_receive_forward_handles = InterleaveScheduleInfo.recv_forward_for_microbatch_id(tensor_shape, config, receive_group, 0)
        if input_tensor is not None:
            input_tensor.record_stream(default_stream)
    input_tensors[0].append(input_tensor)
    wait_receive_backward_handles = None

    # Run warmup forward passes.
    for k in range(num_warmup_microbatches):
        wait_helper(wait_receive_forward_handles)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if k == (total_num_microbatches - 1):
            recv_prev = False

        if recv_prev:
            with torch.cuda.stream(receive_stream):
                input_tensor, wait_receive_forward_handles = InterleaveScheduleInfo.recv_forward_for_microbatch_id(tensor_shape, config, receive_group, k + 1)
                if input_tensor is not None:
                    input_tensor.record_stream(default_stream)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        current_microbatch = InterleaveScheduleInfo.get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(
            k, current_microbatch, checkpoint_activations_microbatch
        )

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None


        with torch.cuda.stream(send_stream):
            send_stream.wait_stream(default_stream)
            send_forward(output_tensor, tensor_shape, config, send_group)
            if output_tensor is not None:
                output_tensor.record_stream(send_stream)

        if (
            k == (num_warmup_microbatches - 1)
            and not forward_only
            and not all_warmup_microbatches
        ):
            input_tensor_grad = None
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False

            output_tensor_grad = None
            if recv_next:
                with torch.cuda.stream(receive_stream):
                    output_tensor_grad, wait_receive_backward_handles = InterleaveScheduleInfo.recv_backward_for_microbatch_id(
                        tensor_shape, config, receive_group, 0
                    )
                    if output_tensor_grad is not None:
                        output_tensor_grad.record_stream(default_stream)
            output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)

        if output_tensor is not None:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.

        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None
        current_microbatch = InterleaveScheduleInfo.get_microbatch_id_in_model_chunk(forward_k, forward=True)
        wait_helper(wait_receive_forward_handles)

        output_tensor = forward_step_helper(
            forward_k, current_microbatch, checkpoint_activations_microbatch
        )

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(forward_k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

        # Last virtual stage no activation tensor to send
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None
        else:
            with torch.cuda.stream(send_stream):
                send_stream.wait_stream(default_stream)

                send_forward(output_tensor,
                             tensor_shape,
                             config,
                             send_group)
                if output_tensor is not None:
                    output_tensor.record_stream(send_stream)

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        next_forward_model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(forward_k + 1, forward=True)
        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        if recv_prev:
            with torch.cuda.stream(receive_stream):
                input_tensor, wait_receive_forward_handles = InterleaveScheduleInfo.recv_forward_for_microbatch_id(tensor_shape,
                                                                                            config,
                                                                                            receive_group,
                                                                                            forward_k + 1)
                if input_tensor is not None:
                    input_tensor.record_stream(default_stream)
        # assert fwd_wait_handles is not None
        if wait_receive_backward_handles is not None:
            wait_helper(wait_receive_backward_handles)

        # Backward pass.
        backward_k = k
        if args.use_nanopipe:
            input_tensor_grad = NanoPipe.backward_step_helper_warper(backward_step_helper, backward_k)
        else:
            input_tensor_grad = backward_step_helper(backward_k)

        backward_model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(backward_k, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

        # First virtual stage no activation gradient tensor to send
        if parallel_state.is_pipeline_first_stage():
            input_tensor_grad = None
        else:
            with torch.cuda.stream(send_stream):
                send_stream.wait_stream(default_stream)
                send_backward(input_tensor_grad,
                             tensor_shape,
                             config,
                             send_group)
                if input_tensor_grad is not None:
                    input_tensor_grad.record_stream(send_stream)

        # Determine if the current virtual stage has an activation gradient tensor to receive
        next_backward_model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(backward_k + 1, forward=False)

        with torch.cuda.stream(receive_stream):
            output_tensor_grad, wait_receive_backward_handles = InterleaveScheduleInfo.recv_backward_for_microbatch_id(tensor_shape,
                                                                                                config,
                                                                                                receive_group,
                                                                                                backward_k + 1
                                                                                                )
            if output_tensor_grad is not None:
                output_tensor_grad.record_stream(default_stream)
        output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        if output_tensor is not None:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            with torch.cuda.stream(receive_stream):
                output_tensor_grad, wait_receive_backward_handles = InterleaveScheduleInfo.recv_backward_for_microbatch_id(tensor_shape,
                                                                                  config,
                                                                                  receive_group,
                                                                                  0)
                if output_tensor_grad is not None:
                    output_tensor_grad.record_stream(default_stream)
            wait_helper(wait_receive_backward_handles)
            output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)

        for k in range(num_microbatches_remaining, total_num_microbatches):
            if wait_receive_backward_handles is not None:
                wait_helper(wait_receive_backward_handles)

            next_backward_model_chunk_id = InterleaveScheduleInfo.get_model_chunk_id(k + 1, forward=False)
            recv_next = True

            if k == (total_num_microbatches - 1):
                recv_next = False

            if recv_next:
                with torch.cuda.stream(receive_stream):
                    output_tensor_grad, wait_receive_backward_handles = InterleaveScheduleInfo.recv_backward_for_microbatch_id(tensor_shape,
                                                                                                        config,
                                                                                                        receive_group,
                                                                                                        k + 1)
                    if output_tensor_grad is not None:
                        output_tensor_grad.record_stream(default_stream)
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
            else:
                output_tensor_grads[next_backward_model_chunk_id].append(None)

            input_tensor_grad = backward_step_helper(k)

            with torch.cuda.stream(send_stream):
                send_stream.wait_stream(default_stream)
                send_backward(
                    input_tensor_grad,
                    tensor_shape,
                    config,
                    send_group
                )
                if input_tensor_grad is not None:
                    input_tensor_grad.record_stream(send_stream)

            if args.use_nanopipe and args.use_nanopipe_swap and k == max(num_microbatches_remaining + 1, (total_num_microbatches + num_microbatches_remaining) // 2):
                NanoPipe.nanopipe_swap_tensors()
        if args.use_nanopipe:
            NanoPipe.nanopipe_cooldown(synchronized_model_chunks, config, model, pipeline_parallel_size)
        # Launch any remaining grad reductions.
        InterleaveScheduleInfo.enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    del input_tensors[:]
    del output_tensors[:]
    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    return forward_data_store