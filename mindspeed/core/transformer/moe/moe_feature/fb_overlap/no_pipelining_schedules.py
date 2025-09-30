# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
from typing import Union, Iterator, List

import torch
from megatron.core.pipeline_parallel.schedules import check_first_val_step
from megatron.core.utils import get_model_config, get_model_type
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.vpp_schedules import forward_step, backward_step, forward_step_func_wrapper


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    # should overide forward step func with extra_block_kwargs passed in
    forward_step_func = forward_step_func_wrapper(forward_step_func)

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    fb_overlap_kwargs = None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                extra_block_kwargs=fb_overlap_kwargs,
            )
            total_num_tokens += num_tokens

            if isinstance(output_tensor, tuple):
                if len(output_tensor) == 2:
                    output_tensor, model_graph = output_tensor
                elif len(output_tensor) == 3:
                    output_tensor, model_graph, pp_comm_output = output_tensor

            if not forward_only:
                output_tensor.backward()  # compute loss backward and detach on final_layernorm
                output_tensor_grad = model_graph[-1].unperm2_graph[1].grad  # get final_layernorm input_tensor.grad

                # prepare FBOverlap kwargs for next forward_step
                fb_overlap_kwargs = {'pp_comm_params': None, 'bwd_pp_comm_params': None,
                                     'bwd_block_output_grad': output_tensor_grad, 'bwd_block_graphs': model_graph}

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
        extra_block_kwargs=fb_overlap_kwargs,
    )
    total_num_tokens += num_tokens

    if isinstance(output_tensor, tuple):
        if len(output_tensor) == 2:
            output_tensor, model_graph = output_tensor
        elif len(output_tensor) == 3:
            output_tensor, model_graph, pp_comm_output = output_tensor

    if not forward_only:
        output_tensor.backward()  # compute loss backward and detach on final_layernorm
        output_tensor_grad = model_graph[-1].unperm2_graph[1].grad  # get final_layernorm input_tensor.grad
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model_graph)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store