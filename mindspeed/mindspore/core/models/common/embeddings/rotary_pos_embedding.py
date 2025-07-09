#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch_npu
import torch
from torch import Tensor
from megatron.training import get_args


def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False,
                              multi_latent_attention: bool = False, mscale: float = 1.0) -> Tensor:
    args = get_args()
    _mscale = mscale
    if hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "yarn":
        _mscale = float(
            yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale)
            / yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale_all_dim)
        )

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = (torch.cos(freqs) * _mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)

    if getattr(args, "use_fused_rotary_pos_emb"):
        mode = 1 if rotary_interleaved else 0
        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
    else:
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    return torch.cat((t, t_pass), dim=-1)
