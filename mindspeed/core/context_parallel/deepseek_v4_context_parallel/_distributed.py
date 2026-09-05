# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Optional, Sequence

import torch
import torch.distributed as dist

from ._utils import _get_cp_size_and_rank, _validate_sequence_tensor_and_positive_length


def _validate_one_hop_tail_coverage(local_shapes, tail_lengths, cp_size):
    if int(cp_size) <= 2:
        return
    unsupported = [
        (int(local_shape[0]), int(tail_length))
        for local_shape, tail_length in zip(local_shapes, tail_lengths)
        if int(tail_length) > int(local_shape[0])
    ]
    if unsupported:
        raise ValueError(
            "DeepSeek V4 CP previous-tail exchange is one-hop: when cp_size > 2, "
            "every local sequence shard must be at least as long as its requested tail. "
            f"Unsupported (local_seq_len, tail_len) pairs: {unsupported}."
        )


class _ReceivePreviousTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_tensor, tail_length, cp_group, cp_global_ranks):
        cp_size, cp_rank = _get_cp_size_and_rank(cp_group)
        ctx.tail_length = tail_length
        ctx.local_shape = tuple(local_tensor.shape)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.cp_global_ranks = cp_global_ranks
        _validate_one_hop_tail_coverage((ctx.local_shape,), (tail_length,), cp_size)

        prev_tail = local_tensor.new_zeros((tail_length,) + tuple(local_tensor.shape[1:]))
        if cp_size == 1:
            return prev_tail

        send_tail = _tail_with_padding(local_tensor, tail_length).contiguous()
        recv_peer = None
        send_peer = None
        if cp_rank > 0:
            recv_peer = cp_global_ranks[cp_rank - 1]
        if cp_rank < cp_size - 1:
            send_peer = cp_global_ranks[cp_rank + 1]
        _send_recv(send_tail, send_peer, prev_tail, recv_peer, cp_group)
        return prev_tail

    @staticmethod
    def backward(ctx, grad_prev_tail):
        grad_local = grad_prev_tail.new_zeros(ctx.local_shape)
        if ctx.cp_size == 1:
            return grad_local, None, None, None

        grad_from_next = grad_prev_tail.new_zeros((ctx.tail_length,) + ctx.local_shape[1:])
        recv_peer = None
        send_peer = None
        if ctx.cp_rank > 0:
            send_peer = ctx.cp_global_ranks[ctx.cp_rank - 1]
        if ctx.cp_rank < ctx.cp_size - 1:
            recv_peer = ctx.cp_global_ranks[ctx.cp_rank + 1]
        _send_recv(grad_prev_tail.contiguous(), send_peer, grad_from_next, recv_peer, ctx.cp_group)

        valid_tail_len = min(ctx.local_shape[0], ctx.tail_length)
        if valid_tail_len > 0:
            grad_local[-valid_tail_len:] = grad_from_next[-valid_tail_len:]
        return grad_local, None, None, None


class _ReceivePreviousTails(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tail_lengths, cp_group, cp_global_ranks, *local_tensors):
        cp_size, cp_rank = _get_cp_size_and_rank(cp_group)
        tail_lengths = tuple(int(length) for length in tail_lengths)
        local_shapes = tuple(tuple(tensor.shape) for tensor in local_tensors)
        tail_shapes = tuple(
            (tail_length,) + local_shape[1:] for tail_length, local_shape in zip(tail_lengths, local_shapes)
        )
        ctx.tail_lengths = tail_lengths
        ctx.local_shapes = local_shapes
        ctx.tail_shapes = tail_shapes
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.cp_global_ranks = cp_global_ranks
        ctx.dtype = local_tensors[0].dtype
        ctx.device = local_tensors[0].device
        ctx.set_materialize_grads(False)
        _validate_one_hop_tail_coverage(local_shapes, tail_lengths, cp_size)

        recv_flat = local_tensors[0].new_zeros(sum(_shape_numel(shape) for shape in tail_shapes))
        if cp_size > 1:
            send_flat = torch.cat(
                [
                    _tail_with_padding(tensor, tail_length).contiguous().reshape(-1)
                    for tensor, tail_length in zip(local_tensors, tail_lengths)
                ],
                dim=0,
            )
            recv_peer = cp_global_ranks[cp_rank - 1] if cp_rank > 0 else None
            send_peer = cp_global_ranks[cp_rank + 1] if cp_rank < cp_size - 1 else None
            _send_recv(send_flat, send_peer, recv_flat, recv_peer, cp_group)

        outputs = []
        offset = 0
        for tail_shape in tail_shapes:
            tail_numel = _shape_numel(tail_shape)
            outputs.append(recv_flat.narrow(0, offset, tail_numel).view(tail_shape).clone())
            offset += tail_numel
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_prev_tails):
        if ctx.cp_size == 1:
            grad_locals = [torch.zeros(shape, dtype=ctx.dtype, device=ctx.device) for shape in ctx.local_shapes]
            return None, None, None, *grad_locals

        grad_parts = []
        for grad_prev_tail, tail_shape in zip(grad_prev_tails, ctx.tail_shapes):
            if grad_prev_tail is None:
                grad_prev_tail = torch.zeros(tail_shape, dtype=ctx.dtype, device=ctx.device)
            grad_parts.append(grad_prev_tail.contiguous().reshape(-1))
        send_flat = torch.cat(grad_parts, dim=0)
        grad_from_next = torch.zeros_like(send_flat)
        send_peer = ctx.cp_global_ranks[ctx.cp_rank - 1] if ctx.cp_rank > 0 else None
        recv_peer = ctx.cp_global_ranks[ctx.cp_rank + 1] if ctx.cp_rank < ctx.cp_size - 1 else None
        _send_recv(send_flat, send_peer, grad_from_next, recv_peer, ctx.cp_group)

        grad_locals = []
        offset = 0
        for local_shape, tail_shape, tail_length in zip(
            ctx.local_shapes,
            ctx.tail_shapes,
            ctx.tail_lengths,
        ):
            tail_numel = _shape_numel(tail_shape)
            received_tail = grad_from_next.narrow(0, offset, tail_numel).view(tail_shape)
            grad_local = torch.zeros(local_shape, dtype=ctx.dtype, device=ctx.device)
            valid_tail_len = min(local_shape[0], tail_length)
            if valid_tail_len > 0:
                grad_local[-valid_tail_len:] = received_tail[-valid_tail_len:]
            grad_locals.append(grad_local)
            offset += tail_numel
        return None, None, None, *grad_locals


def _shape_numel(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _tail_with_padding(local_tensor, tail_length):
    if local_tensor.shape[0] >= tail_length:
        return local_tensor[-tail_length:]
    tail = local_tensor.new_zeros((tail_length,) + tuple(local_tensor.shape[1:]))
    if local_tensor.shape[0] > 0:
        tail[-local_tensor.shape[0] :] = local_tensor
    return tail


def _resolve_previous_tail(local_seq_first, tail_length, previous_tail, seq_dim):
    if previous_tail is None:
        return None
    if not torch.is_tensor(previous_tail):
        raise TypeError("previous_tail must be a torch.Tensor.")
    previous_seq_first = torch.movedim(previous_tail, seq_dim, 0).contiguous()
    expected_shape = (int(tail_length),) + tuple(local_seq_first.shape[1:])
    if tuple(previous_seq_first.shape) != expected_shape:
        raise ValueError(
            "previous_tail shape must match the requested tail and local tensor feature dimensions: "
            f"expected {expected_shape}, got {tuple(previous_seq_first.shape)}."
        )
    if previous_seq_first.device != local_seq_first.device:
        raise ValueError("previous_tail must be on the same device as local_tensor.")
    if previous_seq_first.dtype != local_seq_first.dtype:
        raise ValueError("previous_tail must have the same dtype as local_tensor.")
    return previous_seq_first


def _send_recv(send_tensor, send_peer, recv_tensor, recv_peer, group):
    reqs = []
    if recv_peer is not None:
        reqs.append(_irecv(recv_tensor, recv_peer, group))
    if send_peer is not None:
        reqs.append(_isend(send_tensor, send_peer, group))
    for req in reqs:
        req.wait()


def _irecv(tensor, peer, group):
    if group is None:
        return dist.irecv(tensor, peer)
    return dist.irecv(tensor, peer, group)


def _isend(tensor, peer, group):
    if group is None:
        return dist.isend(tensor, peer)
    return dist.isend(tensor, peer, group)


class _AsyncCollectiveState:
    """Own an asynchronous collective handle until its output is ready."""

    def __init__(self):
        self.work = None
        self.completed = False

    def wait(self):
        if not self.completed and self.work is not None:
            with torch.no_grad():
                self.work.wait()
        self.completed = True


class _AllGatherCompressedKVAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_compressed, cp_group, collective_state):
        cp_size, _ = _get_cp_size_and_rank(cp_group)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.local_first_dim = local_compressed.shape[0]
        ctx.collective_state = collective_state
        if cp_size == 1:
            collective_state.completed = True
            return local_compressed

        gathered = local_compressed.new_empty((cp_size, local_compressed.shape[0]) + tuple(local_compressed.shape[1:]))
        output_views = [gathered[rank] for rank in range(cp_size)]
        collective_state.work = _all_gather_async(
            output_views,
            local_compressed.contiguous(),
            cp_group,
        )
        return gathered.flatten(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        ctx.collective_state.wait()
        if ctx.cp_size == 1:
            return grad_output, None, None
        grad_output = grad_output.contiguous()
        expected_first_dim = ctx.cp_size * ctx.local_first_dim
        if grad_output.shape[0] != expected_first_dim:
            raise RuntimeError(
                "async all-gather compressed-KV gradient has an invalid first dimension: "
                f"expected {expected_first_dim}, got {grad_output.shape[0]}."
            )
        grad_input = grad_output.new_empty((ctx.local_first_dim,) + tuple(grad_output.shape[1:]))
        with torch.no_grad():
            _reduce_scatter_first_dim(grad_input, grad_output, ctx.cp_group)
        return grad_input, None, None


def _all_gather_async(output_tensors, input_tensor, group):
    if group is None:
        return dist.all_gather(output_tensors, input_tensor, async_op=True)
    return dist.all_gather(output_tensors, input_tensor, group=group, async_op=True)


def _reduce_scatter_first_dim(output_tensor, input_tensor, group):
    if hasattr(dist, "reduce_scatter_tensor"):
        if group is None:
            dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
        else:
            dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM, group=group)
        return

    input_chunks = list(input_tensor.chunk(_get_cp_size_and_rank(group)[0], dim=0))
    if group is None:
        dist.reduce_scatter(output_tensor, input_chunks, op=dist.ReduceOp.SUM)
    else:
        dist.reduce_scatter(output_tensor, input_chunks, op=dist.ReduceOp.SUM, group=group)


def _resolve_global_ranks(cp_group, cp_size, cp_global_ranks):
    if cp_size == 1:
        return [0]
    if cp_global_ranks is not None:
        if len(cp_global_ranks) != cp_size:
            raise ValueError("cp_global_ranks length must match the CP group size.")
        return list(cp_global_ranks)
    if cp_group is None:
        return list(range(dist.get_world_size()))
    if hasattr(dist, "get_global_rank"):
        return [dist.get_global_rank(cp_group, group_rank) for group_rank in range(cp_size)]
    raise ValueError("cp_global_ranks is required when the process group cannot expose global ranks.")


def exchange_deepseek_v4_previous_window(
    local_tensor: torch.Tensor,
    window_size: int,
    cp_group=None,
    cp_global_ranks: Optional[Sequence[int]] = None,
    seq_dim: int = 0,
    pad_first_rank: bool = False,
    previous_tail: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prepend the previous CP rank's trailing window to a local sequence shard."""
    _validate_sequence_tensor_and_positive_length(local_tensor, window_size)
    local_seq_first = torch.movedim(local_tensor, seq_dim, 0).contiguous()
    cp_size, cp_rank = _get_cp_size_and_rank(cp_group)
    prev_window = _resolve_previous_tail(local_seq_first, window_size, previous_tail, seq_dim)
    if prev_window is None:
        prev_window = _ReceivePreviousTail.apply(
            local_seq_first,
            window_size,
            cp_group,
            _resolve_global_ranks(cp_group, cp_size, cp_global_ranks),
        )
    if cp_rank == 0 and not pad_first_rank:
        # R14 deadlock fix: keep _ReceivePreviousTail in the autograd graph.
        #
        # Problem: When pad_first_rank=False, rank-0 does not prepend the window
        # (no previous rank exists). If we return local_tensor directly, the
        # _ReceivePreviousTail node is pruned from the autograd graph, so its
        # backward() never runs. This causes deadlock because:
        #   - Rank-1's forward calls _ReceivePreviousTail.backward(), which
        #     posts an irecv expecting rank-0's isend of the window gradient.
        #   - Rank-0 skips backward(), so it never sends the gradient.
        #   - Rank-1 hangs forever waiting for the irecv.
        #
        # Solution: Add a zero-valued dependency on prev_window to
        # force PyTorch to keep _ReceivePreviousTail in the graph. The zero
        # carries no numerical value, but ensures backward() runs on all ranks.
        #
        # Safety: The window gradient received from rank-1 is still correctly
        # folded into local_kv.grad via _ReceivePreviousTail.backward(), because
        # rank-0's tail contributes to both its own output and rank-1's window
        # (see _ReceivePreviousTail.backward implementation at lines 1327-1345).
        #
        # Alternative approaches considered:
        #   1. Always call pad_first_rank=True: rejected because it changes the
        #      tensor shape and breaks downstream shape assumptions.
        #   2. Manual backward hook: rejected because it's harder to maintain
        #      and error-prone across different PyTorch versions.
        #   3. This approach: chosen because it's minimal, correct, and works
        #      with all autograd-aware optimizers.
        return local_tensor + (prev_window.reshape(-1)[0] * 0.0)
    windowed = torch.cat((prev_window, local_seq_first), dim=0)
    return torch.movedim(windowed, 0, seq_dim).contiguous()


def exchange_deepseek_v4_packed_previous_window(
    local_tensor: torch.Tensor,
    window_size: int,
    cu_seqlens,
    local_seq_offset: int,
    cp_group=None,
    cp_global_ranks: Optional[Sequence[int]] = None,
    seq_dim: int = 0,
    pad_first_rank: bool = False,
    previous_tail: Optional[torch.Tensor] = None,
):
    """Build a sample-aware previous-window tensor for packed TND CP shards.

    The local shard is interpreted as a flattened contiguous global token range.
    For every global sample, this function appends a compact ori-KV segment made
    of the same-sample left window plus that sample's local query slice.  Samples
    not intersecting the local shard are retained as zero-length cu_seqlens
    entries so Q/Ori/Cmp TND metadata share the same batch cardinality.
    """
    _validate_sequence_tensor_and_positive_length(local_tensor, window_size)
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for packed window exchange.")
    if local_seq_offset is None:
        raise ValueError("local_seq_offset is required for packed window exchange.")

    local_seq_first = torch.movedim(local_tensor, seq_dim, 0).contiguous()
    local_seq_len = int(local_seq_first.shape[0])
    local_start = int(local_seq_offset)
    local_end = local_start + local_seq_len
    cp_size, cp_rank = _get_cp_size_and_rank(cp_group)

    prev_window = _resolve_previous_tail(local_seq_first, window_size, previous_tail, seq_dim)
    if prev_window is None:
        prev_window = _ReceivePreviousTail.apply(
            local_seq_first,
            window_size,
            cp_group,
            _resolve_global_ranks(cp_group, cp_size, cp_global_ranks),
        )
    source = torch.cat((prev_window, local_seq_first), dim=0)

    valid_prev_len = min(int(window_size), local_start, local_seq_len)
    if cp_rank == 0 and not pad_first_rank:
        valid_prev_len = 0
    available_prev_start = local_start - valid_prev_len

    if torch.is_tensor(cu_seqlens):
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1-D tensor with at least two offsets.")
        cu_host = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    else:
        cu_host = [int(value) for value in cu_seqlens]
        if len(cu_host) < 2:
            raise ValueError("cu_seqlens must be a 1-D tensor with at least two offsets.")
    if cu_host[0] != 0:
        raise ValueError("cu_seqlens must start with 0.")
    if local_start < 0 or local_end > cu_host[-1]:
        raise ValueError(
            "packed local tensor range must be inside cu_seqlens: "
            f"range=[{local_start}, {local_end}), total={cu_host[-1]}."
        )

    chunks = []
    ori_lens = []
    for sample_start, sample_end in zip(cu_host[:-1], cu_host[1:]):
        query_start = max(sample_start, local_start)
        query_end = min(sample_end, local_end)
        if query_end <= query_start:
            ori_lens.append(0)
            continue

        kv_start = max(sample_start, query_start - int(window_size), available_prev_start)
        kv_end = query_end
        if kv_start < local_start:
            source_start = int(window_size) - valid_prev_len + (kv_start - available_prev_start)
        else:
            source_start = int(window_size) + (kv_start - local_start)
        kv_len = kv_end - kv_start
        # The source range is contiguous. Avoiding an index tensor also keeps
        # backward from selecting the out-of-place index_add path.
        chunks.append(source.narrow(0, source_start, kv_len))
        ori_lens.append(kv_len)

    if chunks:
        windowed = torch.cat(chunks, dim=0)
    else:
        windowed = source.new_empty((0,) + tuple(source.shape[1:]))

    lens_tensor = torch.tensor(ori_lens, dtype=torch.int32, device=local_tensor.device)
    cu_ori = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=local_tensor.device),
            torch.cumsum(lens_tensor, dim=0, dtype=torch.int32),
        ),
        dim=0,
    )
    return torch.movedim(windowed, 0, seq_dim).contiguous(), cu_ori


def exchange_deepseek_v4_previous_tails(
    local_tensors: Sequence[torch.Tensor],
    tail_lengths: Sequence[int],
    cp_group=None,
    cp_global_ranks: Optional[Sequence[int]] = None,
    seq_dim: int = 0,
):
    """Exchange differently shaped previous tails through one packed P2P call."""
    local_tensors = tuple(local_tensors)
    tail_lengths = tuple(int(length) for length in tail_lengths)
    if not local_tensors:
        raise ValueError("local_tensors must contain at least one tensor.")
    if len(local_tensors) != len(tail_lengths):
        raise ValueError("tail_lengths must have one entry for every local tensor.")
    for local_tensor, tail_length in zip(local_tensors, tail_lengths):
        _validate_sequence_tensor_and_positive_length(local_tensor, tail_length)

    seq_first_tensors = tuple(torch.movedim(local_tensor, seq_dim, 0).contiguous() for local_tensor in local_tensors)
    first_sequence_tensor = seq_first_tensors[0]
    for tensor in seq_first_tensors[1:]:
        if tensor.device != first_sequence_tensor.device:
            raise ValueError("all local_tensors must be on the same device.")
        if tensor.dtype != first_sequence_tensor.dtype:
            raise ValueError("all local_tensors must have the same dtype.")

    cp_size, _ = _get_cp_size_and_rank(cp_group)
    previous_tails = _ReceivePreviousTails.apply(
        tail_lengths,
        cp_group,
        _resolve_global_ranks(cp_group, cp_size, cp_global_ranks),
        *seq_first_tensors,
    )
    return tuple(torch.movedim(previous_tail, 0, seq_dim).contiguous() for previous_tail in previous_tails)
