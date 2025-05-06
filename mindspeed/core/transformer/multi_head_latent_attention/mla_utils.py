# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import math
from argparse import Namespace

import torch


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(
    num_rotations,
    dim, base=10000,
    max_position_embeddings=2048
):
    return (
        (
            dim * math.log(
                max_position_embeddings / (num_rotations * 2 * math.pi)
            )
        ) / (2 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot,
    high_rot,
    dim,
    base=10000,
    max_position_embeddings=2048
) -> tuple[int, int]:
    low = math.floor(
        yarn_find_correction_dim(
            low_rot, dim, base, max_position_embeddings
        )
    )
    high = math.ceil(
        yarn_find_correction_dim(
            high_rot, dim, base, max_position_embeddings
        )
    )

    # Clamp values just in case
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_, max_, dim) -> torch.Tensor:
    if min_ == max_:
        # Prevent singularity
        max_ += 0.001

    linear_func = (
        (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
    )
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def apply_yarn_scaling(args: Namespace, freqs: torch.Tensor) -> float:

    scaling_factor = args.rope_scaling_factor
    dim = (
        args.qk_rope_head_dim
        if args.multi_head_latent_attention
        else (args.hidden_size // args.num_attention_heads)
    )
    rotary_ratio = (
        args.rotary_base ** (torch.arange(
            0, dim, 2, dtype=torch.float32, device=freqs.device
            ) / dim
        )
    )
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

    inv_freq = (
        freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    )

    return inv_freq
