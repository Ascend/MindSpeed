import torch
import torch_npu
from functools import wraps
from megatron import get_args
from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half


def apply_fused_rotary_pos_emb(t, freqs):

    args = get_args()

    if args.use_fused_rotary_pos_emb:
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def RotaryEmbedding_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        if _args.rotary_base:
            kwargs["rotary_base"] = _args.rotary_base
        fn(self, *args, **kwargs)

    return wrapper
