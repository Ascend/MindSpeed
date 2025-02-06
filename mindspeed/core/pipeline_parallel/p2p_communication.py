# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import List, Optional, Tuple, Union

import torch

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.pipeline_parallel.p2p_communication import _batched_p2p_ops, _p2p_ops
from megatron.core import ModelParallelConfig
from megatron.training import get_args
from mindspeed.utils import get_actual_seq_len, set_actual_seq_len, get_position_ids, set_position_ids
# Types
Shape = Union[List[int], torch.Size]


def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config, tensor_dim: int = 3):
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
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (tensor_dim), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_send_prev=send_prev_shape_tensor,
        tensor_recv_prev=recv_prev_shape_tensor,
        tensor_send_next=send_next_shape_tensor,
        tensor_recv_next=recv_next_shape_tensor,
        group=get_pipeline_model_parallel_group(),
    )

    if len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _communicate(
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        config: ModelParallelConfig,
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

    tensor_recv_prev_func = None
    tensor_recv_next_func = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        tensor_dim = len(tensor_shape) if tensor_shape is not None else 3
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config, tensor_dim,
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
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev_func = create_tensor_recv_prev

    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next_func = create_tensor_recv_next

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    # Each rank can now be part of several different pipeline parallel groups
    # (specifically, this can occur when encoder tensor parallelism != decoder
    # tensor parallelism, and hence a rank in the encoder is going to feed
    # several different decoder ranks. We therefore have to receive or send tensors
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

    reqs = []
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

        reqs.extend(
            p2p_func(
                tensor_send_prev=tensor_send_prev,
                tensor_recv_prev=tensor_recv_prev,
                tensor_send_next=tensor_send_next,
                tensor_recv_next=tensor_recv_next,
                group=group,
                prev_pipeline_rank=pr,
                next_pipeline_rank=nr,
            )
        )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    def _handle_tensor_list(x):
        """This basically handles all the cases that we expect to see. Either the list None,
        or it's a singleton (the usual cases, since most ranks only belong to one pipeline group),
        or everything returned is None, or everything returned is not None, and it has to be summed
        together."""
        if len(x) == 0:
            return None
        if len(x) == 1:
            return x[0]
        if all(xx is None for xx in x):
            return None
        return torch.stack(x, dim=0).sum(dim=0, dtype=torch.float32).to(x[0].dtype)

    tensor_recv_prev = _handle_tensor_list(tensor_recv_prev_list)
    tensor_recv_next = _handle_tensor_list(tensor_recv_next_list)

    return tensor_recv_prev, tensor_recv_next, reqs


def _p2p_ops_eod(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
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

    prev_actual_seq_len = get_actual_seq_len()
    prev_position_ids = get_position_ids()

    tensor_length = None
    length_buffer = None
    args = get_args()
    bsz = args.micro_batch_size
    block_size = args.seq_length // args.context_parallel_size

    if tensor_send_next is not None:
        tensor_length = torch.tensor(prev_actual_seq_len.numel()).npu()
        
    if tensor_recv_prev is not None:
        length_buffer = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
    
    if rank % 2 == 0:
        if tensor_length is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_length, dst=next_pipeline_rank, group=group,
            )
            reqs.append(send_next_req)

        if length_buffer is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=length_buffer, src=prev_pipeline_rank, group=group,
            )
            reqs.append(recv_prev_req)        
    else:
        if length_buffer is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=length_buffer, src=prev_pipeline_rank, group=group,
            )
            reqs.append(recv_prev_req)        

        if tensor_length is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_length, dst=next_pipeline_rank, group=group,
            )
            reqs.append(send_next_req)

    for req in reqs:
        req.wait()
    
    reqs = []

    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            req = torch.distributed.isend(
                tensor=prev_actual_seq_len, dst=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(req)

            req = torch.distributed.isend(
                tensor=prev_position_ids, dst=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(req)

            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            actual_seq_len_buffer = torch.empty([length_buffer.item()], dtype=torch.int64, device=torch.cuda.current_device())

            req = torch.distributed.irecv(
                tensor=actual_seq_len_buffer, src=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(req)
            set_actual_seq_len(actual_seq_len_buffer)

            position_ids_buffer = torch.empty((block_size, bsz), dtype=torch.int64, device=torch.cuda.current_device())
            req = torch.distributed.irecv(
                tensor=position_ids_buffer, src=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            set_position_ids(position_ids_buffer)
            reqs.append(req)

            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            actual_seq_len_buffer = torch.empty([length_buffer.item()], dtype=torch.int64, device=torch.cuda.current_device())

            req = torch.distributed.irecv(
                tensor=actual_seq_len_buffer, src=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(req)
            set_actual_seq_len(actual_seq_len_buffer)

            position_ids_buffer = torch.empty((block_size, bsz), dtype=torch.int64, device=torch.cuda.current_device())
            req = torch.distributed.irecv(
                tensor=position_ids_buffer, src=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            set_position_ids(position_ids_buffer)
            reqs.append(req)

            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            req = torch.distributed.isend(
                tensor=prev_actual_seq_len, dst=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(req)            

            req = torch.distributed.isend(
                tensor=prev_position_ids, dst=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(req)

            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_send_odd_recv_group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_recv_odd_send_group,
            )
            reqs.append(send_prev_req)
    return reqs
    
    
def _p2p_ops_send_recv_overlap(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []
    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_prev,
                prev_pipeline_rank,
                group,
            )
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_prev,
                prev_pipeline_rank,
                group,
            )
            ops.append(recv_prev_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_next,
                next_pipeline_rank,
                group,
            )
            ops.append(send_next_op)
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_next,
                next_pipeline_rank,
                group,
            )
            ops.append(recv_next_op)
    else:
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_next,
                get_pipeline_model_parallel_next_rank(),
                group,
            )
            ops.append(recv_next_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_next,
                get_pipeline_model_parallel_next_rank(),
                group,
            )
            ops.append(send_next_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_prev,
                get_pipeline_model_parallel_prev_rank(),
                group,
            )
            ops.append(recv_prev_op)
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_send_prev,
                get_pipeline_model_parallel_prev_rank(),
                group,
            )
            ops.append(send_prev_op)

    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs
