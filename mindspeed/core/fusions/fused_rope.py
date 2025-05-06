# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
from typing import Optional
import logging

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half
from megatron.training import get_args
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
from mindspeed.core.transformer.multi_head_latent_attention.mla_utils import yarn_get_mscale

try:
    from apex.transformer.functional import (
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )

    HAVE_APPLY_ROPE_FUSION = True
except ImportError:
    HAVE_APPLY_ROPE_FUSION = False



def apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]
        rotary_interleaved (bool, optional): A flag indicating whether to interleave the
            application of sine and cosine components. Defaults to False.
        multi_latent_attention (bool, optional): A flag indicating if multi-latent attention
            mechanism should be applied. If True, it splits the tensor into multiple latent
            spaces before applying RoPE. Defaults to False.
        mscale (float, optional): A scaling factor applied to both the sine and cosine
            components of the positional embeddings. Defaults to 1.0.
    Returns:
        Tensor: The input tensor after applying RoPE
    """
    args = get_args()
    _mscale = mscale
    if getattr(args, "rope_scaling_type", None) == "yarn":
        _mscale = float(
            yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale)
            / yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale_all_dim)
        )
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    args = get_args()
    if getattr(args, "use_fused_rotary_pos_emb", None):
        mode = 1 if rotary_interleaved else 0
        t = npu_rotary_position_embedding(
            t.contiguous(), cos_, sin_, mode
        ).to(t.dtype)
    else:
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    return torch.cat((t, t_pass), dim=-1)


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        #Reset apply_rope_fusion to bypass Megatron core_r0.10.0 check.
        ori_apply_rope_fusion = self.apply_rope_fusion
        self.apply_rope_fusion = False
        print(f"self.apply_rope_fusion:{self.apply_rope_fusion}")
        fn(self)
        self.apply_rope_fusion = ori_apply_rope_fusion
        del ori_apply_rope_fusion

    return wrapper


def apply_rotary_pos_emb(
    t: Tensor, freqs: Tensor, config: TransformerConfig, cu_seqlens: Optional[Tensor] = None, mscale: float = 1.0
):
    """
    Old version for fix rotary_pos_emb in core_r0.10.0.
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    import megatron.core.models.common.embeddings.rope_utils as ru
    logger = logging.getLogger(__name__)
    if config.apply_rope_fusion and not HAVE_APPLY_ROPE_FUSION:
        # setting apply_rope_fusion in config to False so that subsequent queries to this config also return False
        config.apply_rope_fusion = False
        if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
            logger.warning(
                "Setting apply_rope_fusion to false because its implementation"
                " is not included in Apex. Try upgrading to the latest version"
            )
            apply_rotary_pos_emb.printed_fused_warning = True
    if config.apply_rope_fusion:
        if cu_seqlens is None:
            return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    elif cu_seqlens is None:
        return ru._apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale)
    return ru._apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        multi_latent_attention=config.multi_latent_attention,
        mscale=mscale
    )