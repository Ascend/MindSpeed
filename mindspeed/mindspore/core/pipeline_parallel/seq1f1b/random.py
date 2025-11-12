# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
Checkpoint Wrappers for Seq1F1B

This module provides forward and backward wrapper functions for gradient checkpointing
in Seq1F1B training. It handles span_info management across checkpoint boundaries to 
maintain training consistency.
"""

from megatron.training import get_args
from mindspeed.utils import get_actual_seq_len, set_actual_seq_len


def checkpoint_forward_wrapper(fn):
    def wrapper(ctx, run_function, distribute_saved_activations, *args):
        if get_args().seq1f1b_splits > 1:
            ctx.span_info = get_args().span_info
        else:
            ctx.actual_seq_len = get_actual_seq_len()
        return fn(ctx, run_function, distribute_saved_activations, *args)
    return wrapper


def checkpoint_backward_wrapper(fn):
    def wrapper(ctx, *args):
        if get_args().seq1f1b_splits > 1:
            get_args().span_info = ctx.span_info
        else:
            set_actual_seq_len(ctx.actual_seq_len)
        return fn(ctx, *args)
    return wrapper