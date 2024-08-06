from functools import wraps
import torch
from megatron.core.transformer.spec_utils import build_module


def megatron_module_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, config):
        fn(self, config)
        if hasattr(config, 'reset_attention_order') and config.reset_attention_order:
            # Create linear_qkv module before self_attention.
            self.linear_qkv = build_module(torch.nn.GELU)
            # Free memory to avoid memory fragmentation. It will be assigned a real linear function later.
            self.linear_qkv = None
            config.reset_attention_order = False

    return wrapper