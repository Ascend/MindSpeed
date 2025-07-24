"""Adaptor for megatron to enable variable length sequences
in pipeline parallel.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from typing import Iterable, Optional, Tuple

import torch

from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
# _p2p_ops在某些特性中可能被替换
# 需要保证communiacte_shape的时候使用的是原先的版本
from megatron.core.pipeline_parallel.p2p_communication import (
    _batched_p2p_ops as _original_batched_p2p_ops, 
    _p2p_ops as _original_p2p_ops)

from .communicate import Shape, communicate_impl, communicate_shapes_impl


def mindspeed_communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
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

        config (ModelParallelConfig, required): The configuration
            for distributed parallel, which contains
            fields such as tensor_model_parallel_size, etc.

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.
        - reqs: p2p communication requests.
    """
    from megatron.core.pipeline_parallel.p2p_communication import (
        _batched_p2p_ops, 
        _p2p_ops)

    return communicate_impl(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
        get_pipeline_model_parallel_group=get_pipeline_model_parallel_group,
        get_pipeline_model_parallel_next_rank=get_pipeline_model_parallel_next_rank,  # noqa
        get_pipeline_model_parallel_prev_rank=get_pipeline_model_parallel_prev_rank,  # noqa
        batched_p2p_ops=_batched_p2p_ops,
        p2p_ops=_p2p_ops,
        original_batched_p2p_ops=_original_batched_p2p_ops,
        original_p2p_ops=_original_p2p_ops,
        wait_on_reqs=wait_on_reqs,
    )


def mindspeed_commuticate_shapes(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    config: ModelParallelConfig,
    tensor_dim: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        tensor_dim: dimension of the tensor to be sent or be received.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """
    return communicate_shapes_impl(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        config=config,
        get_pipeline_model_parallel_group=get_pipeline_model_parallel_group,
        get_pipeline_model_parallel_next_rank=get_pipeline_model_parallel_next_rank,  # noqa
        get_pipeline_model_parallel_prev_rank=get_pipeline_model_parallel_prev_rank,  # noqa
        batched_p2p_ops=_original_batched_p2p_ops,
        p2p_ops=_original_p2p_ops,
        tensor_dim=tensor_dim,
    )
