# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import math
import torch
from torch import Tensor
from functools import wraps

from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half
from megatron.training import get_args
from megatron.core import parallel_state
from mindspeed.utils import get_position_ids
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

from mindspeed.core.parallel_state import (get_context_parallel_for_hybrid_ulysses_world_size,
                                           get_context_parallel_for_hybrid_ulysses_rank,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank)
from mindspeed.core.context_parallel.utils import get_remapped_seq_order


def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
    )


def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_, max_, dim):
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    args = get_args()
    _mscale = 1.0
    if args.rope_scaling_type == "yarn":
        _mscale = float(
            yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale)
            / yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale_all_dim)
        )

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = (torch.cos(freqs) * _mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)

    if args.use_fused_rotary_pos_emb:
        mode = 1 if rotary_interleaved else 0
        t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
    else:
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    return torch.cat((t, t_pass), dim=-1)


def apply_yarn_scaling(freqs: torch.Tensor):
    args = get_args()

    scaling_factor = args.rope_scaling_factor
    dim = args.qk_rope_head_dim if args.multi_head_latent_attention else (args.hidden_size // args.num_attention_heads)
    rotary_ratio = args.rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=freqs.device) / dim)
    freq_extra = 1.0 / rotary_ratio
    freq_inter = 1.0 / (scaling_factor * rotary_ratio)
    low, high = yarn_find_correction_range(
        args.rope_scaling_beta_fast,
        args.rope_scaling_beta_slow,
        dim,
        args.rotary_base,
        args.rope_scaling_original_max_position_embeddings,
    )

    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
        device=freqs.device, dtype=torch.float32
    )

    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    return inv_freq


def rotary_embedding_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        if _args.rotary_base:
            kwargs["rotary_base"] = _args.rotary_base
        fn(self, *args, **kwargs)
        if hasattr(_args, "rope_scaling_type") and _args.rope_scaling_type == "yarn":
            self.inv_freq = apply_yarn_scaling(self.inv_freq)

    return wrapper


def rotary_forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
    """Forward pass of RoPE embedding.

    Args:
        max_seq_len (int): Maximum size of sequence
        offset (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: Embeddings after applying RoPE.
    """
    seq = (
        torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        + offset
    )

    if self.seq_len_interpolation_factor is not None:
        seq *= 1 / self.seq_len_interpolation_factor

    freqs = torch.outer(seq, self.inv_freq)
    # first part even vector components, second part odd vector components,
    #  2 * dim in dimension size
    if not self.rotary_interleaved:
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
            freqs.shape[0], -1
        )
    # emb [seq_length, .., dim]
    emb = emb[:, None, None, :]
    position_ids = get_position_ids()
    s, b = position_ids.shape
    emb = emb[position_ids.view(-1)].squeeze(1).reshape(s, b, 1, -1)

    if parallel_state.get_context_parallel_world_size() > 1:
        # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
        emb = get_pos_emb_on_this_cp_rank(emb, 0)
    return emb


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    args = get_args()

    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general':
            pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
        else:
            pos_emb = _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.cp_attention_mask_type == 'general':
            pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_cp_general(pos_emb, seq_dim)
        else:
            pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'adaptive_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_adaptive_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_hybrid_adaptive_cp(pos_emb, seq_dim)
    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    pos_emb = pos_emb.chunk(cp_size, dim=seq_dim)[cp_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_hybrid_cp(pos_emb, seq_dim):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()
    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    cp_idx = torch.tensor(
        [r_rank, (2 * r_size - r_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * r_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])

    pos_emb = pos_emb.chunk(u_size, dim=seq_dim)[u_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_hybrid_cp_general(pos_emb, seq_dim):
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()
    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    pos_emb = pos_emb.chunk(r_size, dim=seq_dim)[r_rank]
    pos_emb = pos_emb.chunk(u_size, dim=seq_dim)[u_rank]

    return pos_emb


def _get_pos_emb_on_this_cp_rank_in_adaptive_cp(pos_emd, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    remapped_seq_order = get_remapped_seq_order()
    if remapped_seq_order is not None:
        per = pos_emd.shape[seq_dim] // cp_size
        index = torch.tensor(remapped_seq_order[cp_rank * per:(cp_rank + 1) * per], dtype=torch.int,
                             device=pos_emd.device)
        pos_emd = pos_emd.index_select(seq_dim, index)

    return pos_emd


def _get_pos_emb_on_this_cp_rank_in_hybrid_adaptive_cp(pos_emd, seq_dim):
    ulys_size = get_context_parallel_for_hybrid_ulysses_world_size()
    adap_size = get_context_parallel_for_hybrid_ring_world_size()
    ulys_rank = get_context_parallel_for_hybrid_ulysses_rank()
    adap_rank = get_context_parallel_for_hybrid_ring_rank()

    remapped_seq_order = get_remapped_seq_order()
    if remapped_seq_order is not None:
        per = pos_emd.shape[seq_dim] // adap_size // ulys_size
        which_per = ulys_rank * adap_size + adap_rank
        index = torch.tensor(remapped_seq_order[which_per * per:(which_per + 1) * per], dtype=torch.int,
                             device=pos_emd.device)
        pos_emd = pos_emd.index_select(seq_dim, index)

    return pos_emd

