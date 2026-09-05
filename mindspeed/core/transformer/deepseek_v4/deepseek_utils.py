# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import math
from copy import copy
from functools import lru_cache

import torch
from scipy.linalg import hadamard

from mindspeed.core.context_parallel.deepseek_v4_context_parallel._utils import normalize_cu_seqlens


def max_seqlen_from_cu_seqlens(cu_seqlens):
    if cu_seqlens.numel() <= 1:
        return 0
    lengths = torch.diff(cu_seqlens.to(dtype=torch.long))
    return int(lengths.max().item()) if lengths.numel() else 0


def select_deepseek_v4_cp_packed_seq_params(packed_seq_params, mtp_idx=0):
    if packed_seq_params is None:
        return None
    cu_seqlens_q = getattr(packed_seq_params, "cu_seqlens_q", None)
    if not torch.is_tensor(cu_seqlens_q) or cu_seqlens_q.dim() not in (1, 2):
        raise ValueError("deepseek_v4_cp_algo packed cu_seqlens_q must be one- or two-dimensional.")
    mtp_idx = int(mtp_idx)
    if mtp_idx < 0:
        raise ValueError(f"deepseek_v4_cp_algo mtp_idx must be non-negative, got {mtp_idx}.")
    selected = copy(packed_seq_params)
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        value = getattr(packed_seq_params, name, None)
        if value is not None:
            if not torch.is_tensor(value) or value.dim() not in (1, 2):
                raise ValueError(f"deepseek_v4_cp_algo {name} must be one- or two-dimensional.")
            if value.dim() == 2:
                if mtp_idx >= value.shape[0]:
                    raise ValueError(
                        f"deepseek_v4_cp_algo mtp_idx={mtp_idx} is out of range for {name} with {value.shape[0]} rows."
                    )
                value = value[mtp_idx]
            setattr(selected, name, normalize_cu_seqlens(value, value.device))
    selected_cu_q = selected.cu_seqlens_q
    selected_cu_kv = getattr(selected, "cu_seqlens_kv", None)
    if selected_cu_kv is None:
        selected_cu_kv = selected_cu_q
    selected.max_seqlen_q = max_seqlen_from_cu_seqlens(selected_cu_q)
    selected.max_seqlen_kv = max_seqlen_from_cu_seqlens(selected_cu_kv)
    selected.q_index = None
    selected.kv_index = None
    return selected


def build_deepseek_v4_cp_packed_position_ids(cu_seqlens, positions, device=None):
    if positions is None:
        return None
    if device is None:
        device = positions.device if torch.is_tensor(positions) else None
    cu_seqlens = normalize_cu_seqlens(cu_seqlens, device).to(dtype=torch.long)
    positions = (
        positions.to(device=cu_seqlens.device, dtype=torch.long)
        if torch.is_tensor(positions)
        else torch.tensor(positions, dtype=torch.long, device=cu_seqlens.device)
    )
    original_shape = positions.shape
    flat_positions = positions.reshape(-1)
    if flat_positions.numel() == 0:
        return flat_positions.reshape(original_shape)
    sample_indices = torch.searchsorted(cu_seqlens, flat_positions, right=True) - 1
    safe_sample_indices = sample_indices.clamp(min=0, max=cu_seqlens.numel() - 2)
    sample_starts = cu_seqlens.index_select(0, safe_sample_indices)
    sample_ends = cu_seqlens.index_select(0, safe_sample_indices + 1)
    valid = (
        (sample_indices >= 0)
        & (sample_indices < cu_seqlens.numel() - 1)
        & (flat_positions >= sample_starts)
        & (flat_positions < sample_ends)
    )
    if not bool(valid.all().item()):
        raise ValueError("packed positions must fall inside cu_seqlens sample boundaries.")
    return (flat_positions - sample_starts).reshape(original_shape)


def build_deepseek_v4_cp_local_packed_position_ids(
    packed_seq_params,
    local_seq_len,
    cp_size,
    cp_rank,
    device,
    *,
    position_count=None,
    tp_size=1,
    tp_rank=0,
    sequence_parallel=False,
    get_global=False,
):
    packed_cu = normalize_cu_seqlens(packed_seq_params.cu_seqlens_q, device)
    local_seq_len = int(local_seq_len)
    local_cp_seq_len = local_seq_len * (int(tp_size) if sequence_parallel else 1)
    global_total = int(packed_cu[-1].item())
    if int(cp_size) > 1 and global_total == local_cp_seq_len * int(cp_size):
        cp_local_start = int(cp_rank) * local_cp_seq_len
        cu_for_positions = packed_cu
    elif global_total == local_seq_len:
        local_sample_lens = torch.diff(packed_cu)
        global_sample_lens = local_sample_lens * int(cp_size)
        cu_for_positions = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(global_sample_lens.to(torch.int32), dim=0, dtype=torch.int32),
            ]
        )
        local_count = local_seq_len if position_count is None else int(position_count)
        tp_offset = int(tp_rank) * local_seq_len if sequence_parallel and not get_global else 0
        local_positions = torch.arange(tp_offset, tp_offset + local_count, dtype=torch.long, device=device)
        sample_indices = torch.searchsorted(packed_cu.to(torch.long), local_positions, right=True) - 1
        sample_indices = sample_indices.clamp(min=0, max=packed_cu.numel() - 2)
        local_starts = packed_cu.to(torch.long).index_select(0, sample_indices)
        local_offsets = local_positions - local_starts
        global_starts = cu_for_positions.to(torch.long)[:-1].index_select(0, sample_indices)
        sample_lens = local_sample_lens.to(torch.long).index_select(0, sample_indices)
        global_positions = global_starts + int(cp_rank) * sample_lens + local_offsets
        return build_deepseek_v4_cp_packed_position_ids(cu_for_positions, global_positions, device=device)
    else:
        raise ValueError(
            "packed cu_seqlens_q total must describe the local or global CP tensor: "
            f"cu_total={global_total}, local_seq_len={local_seq_len}, cp_size={cp_size}."
        )

    if position_count is None:
        position_count = local_cp_seq_len if get_global else local_seq_len
    tp_offset = int(tp_rank) * local_seq_len if sequence_parallel else 0
    position_start = (
        cp_local_start if get_global and int(position_count) == local_cp_seq_len else cp_local_start + tp_offset
    )
    positions = torch.arange(position_start, position_start + int(position_count), dtype=torch.long, device=device)
    return build_deepseek_v4_cp_packed_position_ids(cu_for_positions, positions, device=device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2).to(original_dtype)


def apply_rotary_emb_tnd(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(x_complex.size(0), 1, x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    return torch.view_as_real(x_complex * freqs_cis).flatten(-2).to(original_dtype)


def hadamard_transform_ref(x, scale=1.0):
    shape = x.shape
    dim = x.shape[-1]
    padded_dim = 2 ** math.ceil(math.log2(dim))
    flat = x.reshape(-1, dim)
    if dim != padded_dim:
        flat = torch.nn.functional.pad(flat, (0, padded_dim - dim))
    out = torch.nn.functional.linear(flat, get_hadamard_tensor(padded_dim, x.dtype, x.device))
    return (out * scale)[..., :dim].reshape(shape)


@lru_cache(5)
def get_hadamard_tensor(dim_padded, dtype, device):
    return torch.tensor(hadamard(dim_padded, dtype=float), dtype=dtype, device=device)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        hadamard_transform = hadamard_transform_ref
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
