# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import List, Union
import torch

# Types
Shape = Union[List[int], torch.Size]


def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
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
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        rank = get_pipeline_model_parallel_rank()
        if rank % 2 == 0:
            if send_next_shape_tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_next_shape_tensor,
                    get_pipeline_model_parallel_next_rank(),
                )
                ops.append(send_next_op)
            if recv_prev_shape_tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_prev_shape_tensor,
                    get_pipeline_model_parallel_prev_rank(),
                )
                ops.append(recv_prev_op)
            if send_prev_shape_tensor is not None:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_prev_shape_tensor,
                    get_pipeline_model_parallel_prev_rank(),
                )
                ops.append(send_prev_op)
            if recv_next_shape_tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_next_shape_tensor,
                    get_pipeline_model_parallel_next_rank(),
                )
                ops.append(recv_next_op)
        else:
            if recv_prev_shape_tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_prev_shape_tensor,
                    get_pipeline_model_parallel_prev_rank(),
                )
                ops.append(recv_prev_op)
            if send_next_shape_tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_next_shape_tensor,
                    get_pipeline_model_parallel_next_rank(),
                )
                ops.append(send_next_op)
            if recv_next_shape_tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    recv_next_shape_tensor,
                    get_pipeline_model_parallel_next_rank(),
                )
                ops.append(recv_next_op)
            if send_prev_shape_tensor is not None:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend,
                    send_prev_shape_tensor,
                    get_pipeline_model_parallel_prev_rank(),
                )
                ops.append(send_prev_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape