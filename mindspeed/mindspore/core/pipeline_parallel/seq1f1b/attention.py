# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
KV cache management for Seq1F1B attention mechanism.

This module provides wrappers for attention initialization and forward passes to enable
efficient KV cache management in Seq1F1B scheduling.

Functions:
    attention_init_wrapper: Wraps attention init to add KV cache pool
    attention_forward_wrapper: Wraps attention forward to set KV cache context
    core_attention_forward_wrapper: Wraps core attention of dense model (e.g. qwen3) 
    to apply Seq1F1B caching
"""

from functools import wraps
from collections import defaultdict
import torch
from megatron.training import get_args
from .seq1f1b_attn import Seq1F1BCache


def attention_init_wrapper(init_func):
    @wraps(init_func)
    def attention_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        self.kv_cache_pool = defaultdict(dict)
    return attention_init


def attention_forward_wrapper(forward_func):
    @wraps(forward_func)
    def attention_forward(*args, **kwargs):
        self = args[0]
        global_args = get_args()
        global_args.span_info.kv_cache = self.kv_cache_pool[global_args.span_info.micro_batch_idx]
        return forward_func(*args, **kwargs)
    return attention_forward


def core_attention_forward_wrapper(forward_func):
    @wraps(forward_func)
    def core_attention_forward(self, query, key, value, attention_mask, **kwargs):
        global_args = get_args()
        if global_args.seq1f1b_splits > 1 and not global_args.multi_latent_attention:
            key = Seq1F1BCache.apply(key, 'key_cache', global_args.span_info)
            value = Seq1F1BCache.apply(value, 'value_cache', global_args.span_info)
        return forward_func(self, query, key, value, attention_mask, **kwargs)
    return core_attention_forward
