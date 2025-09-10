# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed

from megatron.core import ModelParallelConfig
from megatron.core import parallel_state
from megatron.core.parallel_state import get_pipeline_model_parallel_prev_rank, get_pipeline_model_parallel_next_rank, get_pipeline_model_parallel_rank, get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_group
from mindspeed.core.parallel_state import get_virtual_data_parallel_rank, get_virtual_data_parallel_world_size

# Types
Shape = Union[List[int], Tuple[int], torch.Size]


def _get_pipeline_model_parallel_prev_rank(vdp_rank: int = None):
    if vdp_rank is None:
        vdp_rank = get_virtual_data_parallel_rank()
    vdp_world_size = get_virtual_data_parallel_world_size()
    if vdp_rank < (vdp_world_size // 2):
        return get_pipeline_model_parallel_prev_rank()
    else:
        return get_pipeline_model_parallel_next_rank()


def _get_pipeline_model_parallel_next_rank(vdp_rank: int = None):
    if vdp_rank is None:
        vdp_rank = get_virtual_data_parallel_rank()
    vdp_world_size = get_virtual_data_parallel_world_size()
    if vdp_rank < (vdp_world_size // 2):
        return get_pipeline_model_parallel_next_rank()
    else:
        return get_pipeline_model_parallel_prev_rank()


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    peer_ranks: List[int]
):
    reqs = []
    rank = get_pipeline_model_parallel_rank()
    even_send_odd_recv_group = group
    if get_pipeline_model_parallel_world_size() == 2:
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group
    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=peer_ranks[0],
                group=even_send_odd_recv_group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=peer_ranks[3],
                group=even_recv_odd_send_group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=peer_ranks[1],
                group=even_send_odd_recv_group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=peer_ranks[2],
                group=even_recv_odd_send_group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=peer_ranks[3],
                group=even_send_odd_recv_group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=peer_ranks[0],
                group=even_recv_odd_send_group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=peer_ranks[2],
                group=even_send_odd_recv_group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=peer_ranks[1],
                group=even_recv_odd_send_group,
            )
            reqs.append(send_prev_req)
    return reqs


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    peer_ranks: List[int],
    wait_on_reqs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
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

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    recv_prev_shape = tensor_shape
    recv_next_shape = tensor_shape

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    reqs = _p2p_ops(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=get_pipeline_model_parallel_group(),
        peer_ranks=peer_ranks
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """

    if parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        src = _get_pipeline_model_parallel_prev_rank()
        peer_ranks = [None, None, None, src]
        print(f"[Rank {torch.distributed.get_rank()}]: recv forward from {src}")
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
            peer_ranks=peer_ranks
        )
    return input_tensor


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        src = _get_pipeline_model_parallel_next_rank()
        peer_ranks = [None, None, src, None]
        print(f"[Rank {torch.distributed.get_rank()}]: recv backward from {src}")
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
            peer_ranks=peer_ranks
        )
    return output_tensor_grad


def send_forward(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not parallel_state.is_pipeline_last_stage():
        dst = _get_pipeline_model_parallel_next_rank()
        peer_ranks = [dst, None, None, None]
        print(f"[Rank {torch.distributed.get_rank()}]: send forward to {dst}")
        _, _, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
            peer_ranks=peer_ranks
        )


def send_backward(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not parallel_state.is_pipeline_first_stage():
        dst = _get_pipeline_model_parallel_prev_rank()
        peer_ranks = [None, dst, None, None]
        print(f"[Rank {torch.distributed.get_rank()}]: send backward to {dst}")
        _, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
            peer_ranks=peer_ranks
        )


def send_forward_recv_backward(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig, vdp_rank_1, vdp_rank_2
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline. vdp_rank_1 is always the rank of the current process, vdp_rank_2 is the rank of the next process.

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_last_stage(vdp_rank=vdp_rank_1):
        output_tensor = None
        dst = None
    else:
        dst = _get_pipeline_model_parallel_next_rank(vdp_rank_1)
    if parallel_state.is_pipeline_last_stage(vdp_rank=vdp_rank_2):
        recv_next = False
        src = None
    else:
        recv_next = True
        src = _get_pipeline_model_parallel_next_rank(vdp_rank_2)
    print(f"[Rank {torch.distributed.get_rank()}]: send forward to {dst}, recv backward from {src}")
    
    peer_ranks = [dst, None, src, None]
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
        peer_ranks=peer_ranks
    )
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig, vdp_rank_1, vdp_rank_2
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_first_stage(vdp_rank=vdp_rank_1):
        input_tensor_grad = None
        dst = None
    else:
        dst = _get_pipeline_model_parallel_prev_rank(vdp_rank_1)
    if parallel_state.is_pipeline_first_stage(vdp_rank=vdp_rank_2):
        src = None
        recv_prev = False
    else:
        src = _get_pipeline_model_parallel_prev_rank(vdp_rank_2)
        recv_prev = True
    print(f"[Rank {torch.distributed.get_rank()}]: send backward to {dst}, recv forward from {src}")
    peer_ranks = [None, dst, None, src]
    input_tensor, _, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
        peer_ranks=peer_ranks
    )
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    vdp_rank_1,
    vdp_rank_2
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_last_stage(vdp_rank=vdp_rank_1):
        output_tensor = None
        dst = None
    else:
        dst = _get_pipeline_model_parallel_next_rank(vdp_rank_1)
    if parallel_state.is_pipeline_first_stage(vdp_rank=vdp_rank_2):
        src = None
        recv_prev = False
    else:
        src = _get_pipeline_model_parallel_prev_rank(vdp_rank_2)
        recv_prev = True
    print(f"[Rank {torch.distributed.get_rank()}]: start send forward to {dst}, recv forward from {src}")
    peer_ranks = [dst, None, None, src]
    input_tensor, _, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=True,
        config=config,
        peer_ranks=peer_ranks
    )
    print(f"[Rank {torch.distributed.get_rank()}]: finish send forward to {dst}, recv forward from {src}")
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    vdp_rank_1,
    vdp_rank_2
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_first_stage(vdp_rank=vdp_rank_1):
        input_tensor_grad = None
        dst = None
    else:
        dst = _get_pipeline_model_parallel_prev_rank(vdp_rank_1)
    if parallel_state.is_pipeline_last_stage(vdp_rank=vdp_rank_2):
        recv_next = False
        src = None
    else:
        recv_next = True
        src = _get_pipeline_model_parallel_next_rank(vdp_rank_2)
    print(f"[Rank {torch.distributed.get_rank()}]: send backward to {dst}, recv backward from {src}")
    peer_ranks = [None, dst, src, None]
    _, output_tensor_grad, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=True,  # ovelap_p2p_comm doesn't support yet
        config=config,
        peer_ranks=peer_ranks
    )
    return output_tensor_grad