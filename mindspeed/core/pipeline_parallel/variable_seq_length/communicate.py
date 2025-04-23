"""Define distributed communication between ranks of pipeline parallel,
which can support variable length of sequences.

During communication, all ranks will communicate the tensor shape
it will send/receive in adavance.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch

from .common import Config

# Types
Shape = Union[List[int], torch.Size]


def communicate_shapes_impl(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    config: Config,
    get_pipeline_model_parallel_group: Callable,
    get_pipeline_model_parallel_next_rank: Callable,
    get_pipeline_model_parallel_prev_rank: Callable,
    batched_p2p_ops: Callable,
    p2p_ops: Callable,
    tensor_dim: int = 3,
):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Args:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        config: The configuration for distributed parallel, which contains
            fields such as tensor_model_parallel_size, etc.
        get_pipeline_model_parallel_group: A function to get
            the pipeline model parallel group the caller rank belongs to.
        get_pipeline_model_parallel_next_rank: A function to get t
            he global rank that follows the caller in the pipeline.
        get_pipeline_model_parallel_prev_rank: A function to get the
            lobal rank that preceeds the caller in the pipeline
        batched_p2p_ops: A function to perform batched p2p operations.
        p2p_ops: A function to perform p2p operations.
        tensor_dim: dimension of the tensor to be sent or be received.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (tensor_dim), device=torch.cuda.current_device(), dtype=torch.int64
        )  # type: ignore
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (tensor_dim), device=torch.cuda.current_device(), dtype=torch.int64
        )  # type: ignore
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            # pylint: disable=no-member
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        p2p_func = batched_p2p_ops
    else:
        p2p_func = p2p_ops

    reqs = p2p_func(
        tensor_send_prev=send_prev_shape_tensor,
        tensor_recv_prev=recv_prev_shape_tensor,
        tensor_send_next=send_next_shape_tensor,
        tensor_recv_next=recv_next_shape_tensor,
        group=get_pipeline_model_parallel_group(),
        prev_pipeline_rank=get_pipeline_model_parallel_prev_rank(),
        next_pipeline_rank=get_pipeline_model_parallel_next_rank(),
    )

    if len(reqs) > 0:
        for req in reqs if isinstance(reqs, list) else reqs.values():
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch
        # to not need this
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def communicate_impl(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: Config,
    get_pipeline_model_parallel_group: Callable,
    get_pipeline_model_parallel_next_rank: Callable,
    get_pipeline_model_parallel_prev_rank: Callable,
    batched_p2p_ops: Callable,
    p2p_ops: Callable,
    wait_on_reqs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Iterable[torch.distributed.P2POp]]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Args:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        config: The configuration for distributed parallel, which contains
            fields such as tensor_model_parallel_size, etc.
        get_pipeline_model_parallel_group: A function to get
            the pipeline model parallel group the caller rank belongs to.
        get_pipeline_model_parallel_next_rank: A function to get t
            he global rank that follows the caller in the pipeline.
        get_pipeline_model_parallel_prev_rank: A function to get the
            lobal rank that preceeds the caller in the pipeline
        batched_p2p_ops: A function to perform batched p2p operations.
        p2p_ops: A function to perform p2p operations.

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.
        - reqs: p2p communication requests.
    """

    tensor_recv_prev_func = None
    tensor_recv_next_func = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        tensor_dim = len(tensor_shape) if tensor_shape is not None else 3
        recv_prev_shape, recv_next_shape = communicate_shapes_impl(
            tensor_send_next,
            tensor_send_prev,
            recv_prev,
            recv_next,
            config,
            get_pipeline_model_parallel_group,
            get_pipeline_model_parallel_next_rank,
            get_pipeline_model_parallel_prev_rank,
            batched_p2p_ops,
            p2p_ops,
            tensor_dim,
        )

    def create_tensor_recv_prev():
        return torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    def create_tensor_recv_next():
        return torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError(
                "pipeline_dtype must be provided if recv_prev is True"
            )  # type: ignore
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is "
                "(seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev_func = create_tensor_recv_prev

    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is "
                "(seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next_func = create_tensor_recv_next

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            # pylint: disable=no-member
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = batched_p2p_ops
    else:
        p2p_func = p2p_ops

    # Each rank can now be part of several different pipeline parallel groups
    # (specifically, this can occur when encoder tensor parallelism != decoder
    # tensor parallelism, and hence a rank in the encoder is going to feed
    # several different decoder ranks. We therefore have
    # to receive or send tensors.
    # from several groups. For convenience, I wrap everything into lists.
    pp_group = get_pipeline_model_parallel_group()
    next_rank = get_pipeline_model_parallel_next_rank()
    prev_rank = get_pipeline_model_parallel_prev_rank()
    if not isinstance(pp_group, list):
        pp_group = [pp_group]
        assert not isinstance(next_rank, list)
        next_rank = [next_rank]
        assert not isinstance(prev_rank, list)
        prev_rank = [prev_rank]

    if config.use_ring_exchange_p2p or config.batch_p2p_comm:
        reqs: Union[list, dict] = []
    else:
        reqs = {}
    tensor_recv_prev_list = []
    tensor_recv_next_list = []

    for group, nr, pr in zip(pp_group, next_rank, prev_rank):
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()
            tensor_recv_prev_list.append(tensor_recv_prev)
        else:
            tensor_recv_prev = None

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()
            tensor_recv_next_list.append(tensor_recv_next)
        else:
            tensor_recv_next = None

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=group,
            prev_pipeline_rank=pr,
            next_pipeline_rank=nr,
        )

        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)  # type: ignore
        else:
            reqs.update(p2p_reqs)  # type: ignore

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs if isinstance(reqs, list) else reqs.values():
            req.wait()
        reqs = None  # type: ignore

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch
        # to not need this.
        torch.cuda.synchronize()

    def _handle_tensor_list(x):
        """This basically handles all the cases that we expect to see.
        Either the list None, or it's a singleton
        (the usual cases, since most ranks only belong to one pipeline group),
        or everything returned is None, or everything returned is not None,
        and it has to be summed together.
        """
        if len(x) == 0:
            return None
        if len(x) == 1:
            return x[0]
        if all(xx is None for xx in x):
            return None
        return (
            torch.stack(x, dim=0).sum(dim=0, dtype=torch.float32).to(x[0].dtype)  # noqa
        )  # noqa

    tensor_recv_prev = _handle_tensor_list(tensor_recv_prev_list)
    tensor_recv_next = _handle_tensor_list(tensor_recv_next_list)

    return tensor_recv_prev, tensor_recv_next, reqs
