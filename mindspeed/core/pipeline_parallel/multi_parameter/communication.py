"""Define schedules strategy for multi-parameter parallelism.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from typing import Callable, List, Optional, Union

import torch

from .common import Config


def backward_step_impl(
    input_tensor: Union[torch.Tensor, List[Optional[torch.Tensor]]],
    output_tensor: Union[torch.Tensor, List[torch.Tensor]],
    output_tensor_grad: Union[torch.Tensor, List[torch.Tensor]],
    is_encoder_and_decoder: bool,
    config: Config,
    get_pipeline_model_parallel_world_size: Callable,
    is_pipeline_stage_after_split: Callable,
) -> List[Optional[torch.Tensor]]:
    """Backward step for passed-in model.

    Args:
        input_tensor (Union[torch.Tensor, List[Optional[torch.Tensor]]]):
            input tensor(s) for the backward step.
        output_tensor (Union[torch.Tensor, List[torch.Tensor]]):
            output tensor(s) for the backward step.
        output_tensor_grad (Union[torch.Tensor, List[torch.Tensor]]):
            output tensor gradients for the backward step.
        is_encoder_and_decoder (bool): if the model
            both has encoder and decoder.
        config (object): configuration object of the model.
        get_pipeline_model_parallel_world_size (Callable):
            a function to get the pipeline model parallel world size.
        is_pipeline_stage_after_split (Callable):
            a function to check if the pipeline stage is after split.

    Returns:
        List[Optional[torch.Tensor]]: output tensor of backward step.
    """
    if config.timers is not None:
        config.timers("backward-compute", log_level=2).start()

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
        # The last stage have no input gradients
        # and only one loss is used to backward
        torch.autograd.backward(  # type: ignore
            output_tensor[0], grad_tensors=output_tensor_grad[0]
        )
    else:
        for output, grad in zip(output_tensor, output_tensor_grad):
            if output.requires_grad:
                output_tensors.append(output)
                output_grad_tensors.append(grad)
        torch.autograd.backward(  # type: ignore
            output_tensors, grad_tensors=output_grad_tensors
        )

    # Collect the grad of the input_tensor.
    input_tensor_grad: List[Optional[torch.Tensor]] = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                if x.grad is None:
                    input_tensor_grad.append(
                        torch.zeros_like(
                            x,
                            device=torch.cuda.current_device(),  # type: ignore
                        )
                    )
                else:
                    input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        get_pipeline_model_parallel_world_size() > 1
        and is_pipeline_stage_after_split()
        and is_encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])  # type: ignore
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]  # type: ignore

    if config.timers is not None:
        config.timers("backward-compute").stop()

    return input_tensor_grad


def recv_forward_or_backward(
    tensor_shapes: List[Optional[dict]],
    config: Config,
    recv: Callable,
) -> List[Optional[torch.Tensor]]:
    """receive forward or backward tensor.

    Args:
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward or backward step.
        config (object): configuration object of the model.
        recv_forward_ (Callable):
            a p2p communication function to receive
            the forward or backward tensor.

    Returns:
        List[Optional[torch.Tensor]]: the input tensors
            for the forward or backward step.
    """
    tensors: List[Optional[torch.Tensor]] = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            tensors.append(None)
        else:
            config.pipeline_dtype = tensor_shape["dtype"]
            tensors.append(recv(tensor_shape["shape"], config))
    return tensors


def recv_forwrard_impl(
    tensor_shapes: List[Optional[dict]],
    config: Config,
    recv_forward_: Callable,
) -> List[Optional[torch.Tensor]]:
    """Receive forward tensor.

    Args:
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward step.
        config (object): configuration object of the model.
        recv_forward_ (Callable):
            a p2p communication function to receive
            the forward tensor.

    Returns:
        List[Optional[torch.Tensor]]: the input tensors
            for the forward step.
    """
    return recv_forward_or_backward(tensor_shapes, config, recv_forward_)


def recv_backward_impl(
    tensor_shapes: List[Optional[dict]],
    config: Config,
    recv_backward_: Callable,
) -> List[Optional[torch.Tensor]]:
    """Receive backward tensor.

    Args:
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the backward step.
        config (object): configuration object of the model.
        recv_backward_ (Callable):
            a p2p communication function to receive
            the backward tensor.

    Returns:
        List[Optional[torch.Tensor]]: the input tensors
            for the backward step.
    """
    return recv_forward_or_backward(tensor_shapes, config, recv_backward_)


def send_forward_or_backward(
    tensors: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
    send_forward_or_backward_: Callable,
):
    """Send forward or backward tensor.

    Args:
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward or backward step.
        config (object): configuration object of the model.
        send_forward_or_backward_ (Callable):
            a p2p communication function to send
            the forward or backward tensor.
    """
    if tensors is None:
        tensors = [None] * len(tensor_shapes)
    if not isinstance(tensors, list):
        tensors = [tensors]
    for tensor, tensor_shape in zip(tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        config.pipeline_dtype = tensor_shape["dtype"]
        send_forward_or_backward_(tensor, config)


def send_forward_impl(
    output_tensors: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
    send_forward_: Callable,
):
    """Send forward tensor.

    Args:
        output_tensors (List[Optional[torch.Tensor]]):
            the output tensors for the forward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward step.
        config (object): configuration object of the model.
        send_forward_ (Callable):
            a p2p communication function to send
            the forward tensor.
    """
    return send_forward_or_backward(
        output_tensors,
        tensor_shapes,
        config,
        send_forward_,
    )


def send_backward_impl(
    input_tensor_grads: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
    send_backward_: Callable,
):
    """Send backward tensor.

    Args:
        input_tensor_grads (List[Optional[torch.Tensor]]):
            the input grad tensors for the backward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the backward step.
        config (object): configuration object of the model.
        send_backward_ (Callable):
            a p2p communication function to send
            the backward tensor.
    """
    return send_forward_or_backward(
        input_tensor_grads,
        tensor_shapes,
        config,
        send_backward_,
    )


def send_forward_and_backward(
    tensors: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
    send_forward_and_backward_: Callable,
) -> List[Optional[torch.Tensor]]:
    """Send forward and backward tensor.

    Args:
        tensors (List[Optional[torch.Tensor]]):
            the input tensors for the forward and backward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward and backward step.
        config (object): configuration object of the model.
        send_forward_or_backward_ (Callable):
            a p2p communication function to send
            the forward and backward tensor.

    Returns:
        List[Optional[torch.Tensor]]: the output tensors
            for the backward and forwardstep.
    """
    if not isinstance(tensors, list):
        tensors = [None] * len(tensor_shapes)
    results: List[Optional[torch.Tensor]] = []
    for tensor, tensor_shape in zip(tensors, tensor_shapes):
        if tensor_shape is None:
            results.append(None)
            continue
        config.pipeline_dtype = tensor_shape["dtype"]
        ret = send_forward_and_backward_(tensor, tensor_shape["shape"], config)
        results.append(ret)
    return results


def send_forward_recv_backward_impl(
    output_tensors: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
    send_forward_recv_backward_: Callable,
) -> List[Optional[torch.Tensor]]:
    """Send forward and receive backward tensor.

    Args:
        output_tensors (List[Optional[torch.Tensor]]):
            the output tensors for the forward step.
        tensor_shapes (List[Optional[dict]]):
            tensor shapes for the forward step.
        config (object): configuration object of the model.
        send_forward_recv_backward_ (Callable):
            a p2p communication function to send
            the forward tensor and receive the backward tensor.

    Returns:
        List[Optional[torch.Tensor]]: the output tensors
            for the backward step.
    """
    return send_forward_and_backward(
        output_tensors, tensor_shapes, config, send_forward_recv_backward_
    )


def send_backward_recv_forward_impl(
    input_tensor_grads: List[Optional[torch.Tensor]],
    tensor_shapes: List[Optional[dict]],
    config: Config,
    send_backward_recv_forward_: Callable,
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
    return send_forward_and_backward(
        input_tensor_grads,
        tensor_shapes,
        config,
        send_backward_recv_forward_,
    )
