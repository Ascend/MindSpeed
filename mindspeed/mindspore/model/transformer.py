# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from functools import wraps
from megatron.training import get_args
import mindtorch.torch as torch
try:
    from einops import rearrange
except ImportError:
    rearrange = None


def generate_attention_mask(args, attention_mask, device):
    """generate attention mask."""

    if attention_mask is None:
        attention_mask = (torch.tril(
            torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=torch.int, device=device),
            diagonal=-(args.pre_tockens + 1)) \
                          + torch.triu(
                    torch.ones([args.micro_batch_size, 1, args.seq_length, args.seq_length], dtype=torch.int,
                               device=device), diagonal=args.next_tockens + 1))
        attention_mask = attention_mask.to(torch.bool)
    return attention_mask


def parallel_transformer_forward_wrapper(fn):
    """wrapper parallel transformer forward method"""

    @wraps(fn)
    def wrapper(self, hidden_states, attention_mask, **kwargs):
        args = get_args()
        attention_mask = generate_attention_mask(args, attention_mask, hidden_states.device)
        return fn(self, hidden_states, attention_mask, **kwargs)

    return wrapper
