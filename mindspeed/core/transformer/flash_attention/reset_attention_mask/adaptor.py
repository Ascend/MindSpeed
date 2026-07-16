# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps
import inspect
import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams

from mindspeed.utils import get_position_ids, set_position_ids
from mindspeed.core.context_parallel.get_batch_utils import get_actual_seq_len, set_actual_seq_len
from mindspeed.core.fusions.fused_rope import apply_rotary_pos_emb_bshd


def attention_forward_wrapper(fn):
    """Adapt Megatron's dummy-batch THD contract at the Attention boundary.

    MindSpeed master keeps ``[s, b, h]`` through the model and flattens only
    for attention.  Megatron 0.17's THD path expects a dummy batch of one.
    Folding here, instead of in GPTModel, preserves PP/MTP/loss shapes and can
    be migrated independently when the Megatron 0.18 API adaptation lands.
    """

    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        hidden_states = bound.arguments.get('hidden_states')
        packed_seq_params = bound.arguments.get('packed_seq_params')
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if not is_thd or hidden_states is None or hidden_states.dim() != 3 or hidden_states.shape[1] <= 1:
            return fn(*args, **kwargs)

        seq_length, batch_size, hidden_size = hidden_states.shape
        bound.arguments['hidden_states'] = (
            hidden_states.transpose(0, 1).reshape(batch_size * seq_length, 1, hidden_size).contiguous()
        )

        key_value_states = bound.arguments.get('key_value_states')
        if key_value_states is not None:
            if key_value_states.dim() != 3 or key_value_states.shape[1] != batch_size:
                raise AssertionError('EOD cross-attention key/value batch shape does not match hidden_states.')
            bound.arguments['key_value_states'] = (
                key_value_states.transpose(0, 1)
                .reshape(batch_size * key_value_states.shape[0], 1, key_value_states.shape[2])
                .contiguous()
            )

        position_ids = bound.arguments.get('position_ids')
        if position_ids is not None:
            bound.arguments['position_ids'] = position_ids.reshape(1, -1).contiguous()

        saved_position_ids = get_position_ids()
        if saved_position_ids is not None:
            if saved_position_ids.shape != (seq_length, batch_size):
                raise AssertionError(
                    'EOD position_ids must match Attention [sequence, batch] shape; '
                    f'got {tuple(saved_position_ids.shape)} and {(seq_length, batch_size)}.'
                )
            set_position_ids(saved_position_ids.transpose(0, 1).reshape(batch_size * seq_length, 1).contiguous())

        try:
            output, bias = fn(*bound.args, **bound.kwargs)
        finally:
            if saved_position_ids is not None:
                set_position_ids(saved_position_ids)

        if output.dim() != 3 or output.shape[0] != batch_size * seq_length or output.shape[1] != 1:
            raise AssertionError('Megatron THD Attention must return [batch * sequence, 1, hidden] before EOD restore.')
        output = output.reshape(batch_size, seq_length, output.shape[-1]).transpose(0, 1).contiguous()
        return output, bias

    return wrapper


def p2p_communicate_eod_wrapper(fn):
    """Send EOD metadata alongside forward pipeline activations.

    This wraps Megatron 0.17's communicator instead of replacing its batched
    and unbatched request builders, so native request container and wait
    semantics remain unchanged.
    """

    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        communicator = bound.arguments['self']
        tensor_send_next = bound.arguments.get('tensor_send_next')
        recv_prev = bool(bound.arguments.get('recv_prev'))
        send_metadata = tensor_send_next is not None
        recv_metadata = recv_prev

        if send_metadata or recv_metadata:
            group = communicator.pp_group
            device = torch.device('cuda', torch.cuda.current_device())
            header_send = None
            header_recv = None

            if send_metadata:
                actual_seq_len = get_actual_seq_len()
                position_ids = get_position_ids()
                if actual_seq_len is None or position_ids is None:
                    raise AssertionError('EOD PP send requires actual_seq_len and position_ids.')
                if position_ids.dim() != 2:
                    raise AssertionError('EOD PP position_ids metadata must be two-dimensional.')
                actual_seq_len = actual_seq_len.to(device=device, dtype=torch.int64).contiguous()
                position_ids = position_ids.to(device=device, dtype=torch.int64).contiguous()
                header_send = torch.tensor(
                    [actual_seq_len.numel(), position_ids.shape[0], position_ids.shape[1]],
                    dtype=torch.int64,
                    device=device,
                )

            if recv_metadata:
                header_recv = torch.empty(3, dtype=torch.int64, device=device)

            header_ops = []
            if header_send is not None:
                header_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        header_send,
                        communicator.next_rank,
                        group,
                    )
                )
            if header_recv is not None:
                header_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.irecv,
                        header_recv,
                        communicator.prev_rank,
                        group,
                    )
                )
            for request in torch.distributed.batch_isend_irecv(header_ops):
                request.wait()

            actual_seq_len_recv = None
            position_ids_recv = None
            if header_recv is not None:
                actual_seq_len_recv = torch.empty(int(header_recv[0].item()), dtype=torch.int64, device=device)
                position_ids_recv = torch.empty(
                    (int(header_recv[1].item()), int(header_recv[2].item())),
                    dtype=torch.int64,
                    device=device,
                )

            metadata_ops = []
            if send_metadata:
                metadata_ops.extend(
                    [
                        torch.distributed.P2POp(
                            torch.distributed.isend,
                            actual_seq_len,
                            communicator.next_rank,
                            group,
                        ),
                        torch.distributed.P2POp(
                            torch.distributed.isend,
                            position_ids,
                            communicator.next_rank,
                            group,
                        ),
                    ]
                )
            if recv_metadata:
                metadata_ops.extend(
                    [
                        torch.distributed.P2POp(
                            torch.distributed.irecv,
                            actual_seq_len_recv,
                            communicator.prev_rank,
                            group,
                        ),
                        torch.distributed.P2POp(
                            torch.distributed.irecv,
                            position_ids_recv,
                            communicator.prev_rank,
                            group,
                        ),
                    ]
                )
            for request in torch.distributed.batch_isend_irecv(metadata_ops):
                request.wait()

            if recv_metadata:
                set_actual_seq_len(actual_seq_len_recv)
                set_position_ids(position_ids_recv)

        return fn(*args, **kwargs)

    return wrapper


def gpt_forward_wrapper(fn):
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        actual_seq_len = get_actual_seq_len()
        if actual_seq_len is None:
            raise AssertionError('reset-attention-mask needs actual_seq_len for THD EOD path.')

        if bound.arguments.get('packed_seq_params') is not None:
            raise AssertionError(
                'reset-attention-mask EOD metadata cannot be combined with another packed_seq_params source.'
            )

        position_ids = bound.arguments.get('position_ids')
        if position_ids is not None:
            set_position_ids(position_ids.transpose(0, 1).contiguous())

        packed_seq_params = PackedSeqParams(qkv_format='thd', cu_seqlens_q=actual_seq_len, cu_seqlens_kv=actual_seq_len)

        actual_seq_len_list = actual_seq_len.tolist()
        max_actual_seq_len = actual_seq_len_list[0]
        for i in range(1, len(actual_seq_len_list)):
            max_actual_seq_len = max(max_actual_seq_len, actual_seq_len_list[i] - actual_seq_len_list[i - 1])
        packed_seq_params.max_seqlen_q = max_actual_seq_len
        packed_seq_params.max_seqlen_kv = max_actual_seq_len

        packed_seq_params.position_ids = get_position_ids()

        bound.arguments['packed_seq_params'] = packed_seq_params
        return fn(*bound.args, **bound.kwargs)

    return wrapper


def rotary_seq_len_eod_wrapper(fn):
    """Size the THD RoPE table from EOD positions, not local CP token count.

    Ring P2P places the first and last chunk of every document on one rank.
    The latter retains its document-relative position ids, which can be larger
    than the rank-local packed sequence length. Megatron's native THD branch
    uses ``max_seqlen`` for the RoPE table; use the actual EOD positions here
    so indexing remains valid after the P2P layout transformation.
    """
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        packed_seq_params = bound.arguments.get('packed_seq_params')
        position_ids = get_position_ids()
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd' and position_ids is not None:
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise AssertionError(f'THD EOD RoPE position_ids must use an integral dtype, got {position_ids.dtype}.')
            if position_ids.numel() == 0:
                raise AssertionError('THD EOD RoPE position_ids must not be empty.')
            max_position_id = int(position_ids.max().item())
            if max_position_id < 0:
                raise AssertionError('THD EOD RoPE position_ids must be non-negative.')
            return max_position_id + 1
        return fn(*args, **kwargs)

    return wrapper


def apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
        cp_group (torch.distributed.ProcessGroup): The context parallel group (unused in this path).

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    # 从全局变量 _POSITION_IDS 获取 position_ids（EOD 边界重置），
    # 解除对 cu_seqlens 参数的依赖，使 Megatron 原生 MLA 的
    # get_query_key_value_tensors 正常工作。
    position_ids = get_position_ids()
    if position_ids is None:
        raise AssertionError('reset-attention-mask needs position_ids for THD EOD RoPE path.')
    if t.dim() == 3:
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise AssertionError(f'THD EOD RoPE position_ids must use an integral dtype, got {position_ids.dtype}.')
        flat_position_ids = position_ids.reshape(-1).to(device=freqs.device, dtype=torch.long, non_blocking=True)
        if flat_position_ids.numel() != t.shape[0]:
            raise AssertionError(
                f'THD RoPE needs position_ids numel ({flat_position_ids.numel()}) to match token count ({t.shape[0]}).'
            )
        if torch.any(flat_position_ids < 0) or torch.any(flat_position_ids >= freqs.shape[0]):
            raise AssertionError(
                f'THD EOD RoPE position_ids exceed the generated frequency table: freqs length={freqs.shape[0]}.'
            )
        freqs = freqs[flat_position_ids]
        return apply_rotary_pos_emb_bshd(
            t.unsqueeze(1), freqs, rotary_interleaved, multi_latent_attention, mscale
        ).squeeze(1)

    block_size, bsz = position_ids.shape
    freqs = freqs[position_ids.view(-1)].reshape(block_size, bsz, 1, -1)

    return apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved, multi_latent_attention, mscale)
