"""This module aims to make adaptor for megatron.

Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from functools import wraps
from typing import List, Optional, Union

import torch

from megatron.core.enums import ModelType
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_world_size,
    is_pipeline_stage_after_split)
from megatron.core.pipeline_parallel.p2p_communication import \
    recv_backward as recv_backward_
from megatron.core.pipeline_parallel.p2p_communication import \
    recv_forward as recv_forward_
from megatron.core.pipeline_parallel.p2p_communication import \
    send_backward as send_backward_
from megatron.core.pipeline_parallel.p2p_communication import \
    send_backward_recv_forward as send_backward_recv_forward_
from megatron.core.pipeline_parallel.p2p_communication import \
    send_forward as send_forward_
from megatron.core.pipeline_parallel.p2p_communication import \
    send_forward_recv_backward as send_forward_recv_backward_
from mindspeed.core.pipeline_parallel.multiparameter_schedules import \
    forward_backward_pipelining_with_interleaving

from .common import Config
from .communication import (backward_step_impl, recv_backward_impl,
                            recv_forwrard_impl, send_backward_impl,
                            send_backward_recv_forward_impl, send_forward_impl,
                            send_forward_recv_backward_impl)


def get_tensor_shapes_wrapper(fn):
    """A decorator for function that get tensor shapes
    between two pipeline stage.
    """

    @wraps(fn)
    def wrapper(*arg, **kwargs):
        """Get tensor shapes for multi parameter situation.

        Returns:
            List[dict]: tensor shapes for multi parameter situation.
        """
        config = kwargs["config"]
        return config.pipeline_tensor_shapes

    return wrapper


def forward_step_wrapper(fn):
    """The fn is to forward_step for passed-in model,
        it have args below.

    Args:
        fn (Callable): A function to forward step for passed-in model.
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output
            a tuple of two elements:

            1. The output object from the forward step. This output object
                needs to be atensor or some kind of collection of tensors.
                The only hard requirementfor this object is that
                it needs to be acceptible as input into the second function.
            2. A function to reduce (optionally) the output from
                the forward step. This could be a reduction over the loss
                from the model, it could be a function that grabs the output
                from the model and reformats, it could be a function that just
                passes through the model output.
                This function must have one of the following patterns,
                and depending on the pattern different things
                happen internally:

                    a. A tuple of reduced loss and some other data.
                        Note that in this case the first argument is divided
                        by the number of global microbatches,
                        assuming it is a loss, so that the loss is stable
                        as a function of the number of devices the step
                        is split across.
                    b. A triple of reduced loss, number of tokens,
                        and some other data. This is similar to case (a),
                        but the loss is further averaged across the
                        number of tokens in the batch.
                        If the user is not already averaging across the number
                        of tokens, this pattern is useful to use.
                    c. Any arbitrary data the user wants
                        (eg a dictionary of tensors, a list of tensors,
                        etc in the case of inference). To trigger case 3
                        you need to specify `collect_non_loss_data=True`
                        and you may also want to specify `forward_only=True`
                        in the call to the parent forward_backward function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then
            this will store only the final dimension of the output,
            for example the metadata output by the loss function.
            If you go down the path of 2.c then this
            will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output
            from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tuple[List[torch.Tensor], int]: output tensor and number of tokens.
    """

    @wraps(fn)
    def wrapper(*arg, **kwargs):
        output_tensor, num_tokens = fn(*arg, **kwargs)
        if len(output_tensor) > 0 and isinstance(output_tensor[0], list):
            return output_tensor[0], num_tokens
        else:
            return output_tensor, num_tokens

    return wrapper


def mindspeed_backward_step(
    input_tensor: Union[torch.Tensor, List[Optional[torch.Tensor]]],
    output_tensor: Union[torch.Tensor, List[torch.Tensor]],
    output_tensor_grad: Union[torch.Tensor, List[torch.Tensor]],
    model_type: bool,
    config: Config,
) -> List[Optional[torch.Tensor]]:
    """Backward step for passed-in model.

    Args:
        input_tensor (Union[torch.Tensor, List[Optional[torch.Tensor]]]):
            input tensor(s) for the backward step.
        output_tensor (Union[torch.Tensor, List[torch.Tensor]]):
            output tensor(s) for the backward step.
        output_tensor_grad (Union[torch.Tensor, List[torch.Tensor]]):
            output tensor gradients for the backward step.
        model_type (bool): type of the model.
        config (object): configuration object of the model.

    Returns:
        List[Optional[torch.Tensor]]: output tensor of backward step.
    """
    is_encoder_and_decoder = model_type == ModelType.encoder_and_decoder
    return backward_step_impl(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        output_tensor_grad=output_tensor_grad,
        is_encoder_and_decoder=is_encoder_and_decoder,
        config=config,
        get_pipeline_model_parallel_world_size=get_pipeline_model_parallel_world_size,  # noqa
        is_pipeline_stage_after_split=is_pipeline_stage_after_split,
    )


def mindspeed_recv_forward(
    tensor_shapes: List[Optional[dict]],
    config: Config,
) -> List[Optional[torch.Tensor]]:
    """Receive forward tensor.

    Args:
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward step.
        config (object): configuration object of the model.

    Returns:
        List[Optional[torch.Tensor]]: the input tensors
            for the forward step.
    """
    return recv_forwrard_impl(
        tensor_shapes=tensor_shapes,
        config=config,
        recv_forward_=recv_forward_,
    )


def mindspeed_recv_backward(
    tensor_shapes: List[Optional[dict]],
    config: Config,
) -> List[Optional[torch.Tensor]]:
    """Receive forward tensor.

    Args:
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the backward step.
        config (object): configuration object of the model.

    Returns:
        List[Optional[torch.Tensor]]: the input tensors
            for the forward step.
    """
    return recv_backward_impl(
        tensor_shapes=tensor_shapes,
        config=config,
        recv_backward_=recv_backward_,
    )


def mindspeed_send_forward(
    output_tensors: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
):
    """Send forward tensor.

    Args:
        output_tensors (List[Optional[torch.Tensor]]):
            the output tensors for the forward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward step.
        config (object): configuration object of the model.
    """
    return send_forward_impl(
        output_tensors=output_tensors,
        tensor_shapes=tensor_shapes,
        config=config,
        send_forward_=send_forward_,
    )


def mindspeed_send_backward(
    input_tensor_grads: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
):
    """Send backward tensor.

    Args:
        input_tensor_grads (List[Optional[torch.Tensor]]):
            the input grad tensors for the backward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the backward step.
        config (object): configuration object of the model.
    """
    return send_backward_impl(
        input_tensor_grads=input_tensor_grads,
        tensor_shapes=tensor_shapes,
        config=config,
        send_backward_=send_backward_,
    )


def mindspeed_send_forward_recv_backward(
    output_tensors: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
):
    """Send forward and receive backward tensor.

    Args:
        output_tensors (List[Optional[torch.Tensor]]):
            the output tensors for the forward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward step.
        config (object): configuration object of the model.

    Returns:
        List[Optional[torch.Tensor]]: the output tensors
            for the backward step.
    """
    return send_forward_recv_backward_impl(
        output_tensors=output_tensors,
        tensor_shapes=tensor_shapes,
        config=config,
        send_forward_recv_backward_=send_forward_recv_backward_,
    )


def mindspeed_send_backward_recv_forward(
    input_tensor_grads: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
) -> List[Optional[torch.Tensor]]:
    """Send backward and receive forward tensor.

    Args:
        input_tensor_grads (List[Optional[torch.Tensor]]):
            the input grad tensors for the backward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the backward step.
        config (object): configuration object of the model.
        send_backward_recv_forward_ (Callable):
            a p2p communication function to send
            the backward tensor and receive the forward tensor.

    Returns:
        List[Optional[torch.Tensor]]: the output tensors
            for the forward step.
    """
    return send_backward_recv_forward_impl(
        input_tensor_grads=input_tensor_grads,
        tensor_shapes=tensor_shapes,
        config=config,
        send_backward_recv_forward_=send_backward_recv_forward_,
    )


def get_forward_backward_func_wrapper(fn):
    """Get forward and backward function for multi parameter model.

    Returns:
        Callable: A fun that run interleaved 1F1B schedule
        (model split into model chunks), with communication between
        pipeline stages as needed for multi parameter.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if (
            get_pipeline_model_parallel_world_size() > 1
            and get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            return forward_backward_pipelining_with_interleaving
        return fn(*args, **kwargs)

    return wrapper


def core_transformer_config_from_args_wrapper(fn):
    """A decorator for transformer config."""

    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        if args.use_multiparameter_pipeline_model_parallel:
            config.deallocate_pipeline_outputs = False

    return wrapper
