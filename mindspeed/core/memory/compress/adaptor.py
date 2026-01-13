# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import uuid
from typing import Callable, List, Optional  # codecheck_ignore
from functools import wraps, partial

import torch
from megatron.training import get_args

from mindspeed.core.memory.compress.compress_optimizer import compress_optimizer_step_impl
from mindspeed.core.memory.compress.compress_activation import CompressHook, GlobalContext, GlobalContextConfig


def compress_optimizer_step(self, closure=None):
    with torch.no_grad():
        loss = compress_optimizer_step_impl(self, closure)
    return loss


def layer_forward_wrapper(forward) -> Callable:
    """ Main enterence of adaptive activation compression.
    """
    @wraps(forward)
    def wrapper(self, *args, **kwargs):
        if self.layer_number not in get_args().compress_activation:
            return forward(self, *args, **kwargs)
        ctx = get_global_context()
        ctx.statistic = get_statistic()
        absolute_order = get_absolute_order(ctx)
        if not hasattr(self, "layer_uuid"):
            self.layer_uuid = uuid.uuid1()
        order_layer_uuid = (self.layer_uuid, absolute_order)
        ctx.pack_start(order_layer_uuid)
        with CompressHook(order_layer_uuid, ctx):
            result = forward(self, *args, **kwargs)
        if result is None or result[0].grad_fn is None:
            raise RuntimeError(
                "Result is None or grad_fn is missing."
                "Ensure the model is in training mode and gradients are enabled."
            )
        result[0].register_hook(partial(backward_start, order_layer_uuid))
        return result
    return wrapper


def backward_start(order_layer_uuid, *unused) -> None:
    get_global_context().unpack_start(order_layer_uuid)


def get_global_context() -> GlobalContext:
    args = get_args()
    if not hasattr(args, "global_context"):
        filter_funcs = [filter_func]
        async_funcs = ["matmul", "allgather", "all2all"]
        config = GlobalContextConfig(
            get_moe_average_token_num(),
            get_args().num_experts,
            filter_funcs, 
            async_funcs
        )
        setattr(args, "global_context", GlobalContext(config))
    return args.global_context


def get_statistic() -> int:
    """ Determine whether each encoding requires re-statistical analysis of the PDF.
    """
    increment = get_args().curr_iteration - get_args().iteration
    return increment <= 3 or increment % 100 == 0


def get_moe_average_token_num() -> int:
    """ Calculate the average number of tokens allocated per card 
    for the MoE model based on configuration.
    """
    args = get_args()
    return args.micro_batch_size * args.seq_length * args.tensor_model_parallel_size * args.moe_router_topk


def get_absolute_order(ctx: GlobalContext) -> int:
    is_first_step = (get_args().curr_iteration - get_args().iteration) == 0
    return ctx.get_absolute_order(is_first_step)


def filter_func(tensor: torch.Tensor) -> bool:
    """ Exclude tensors that do not require compression.
    """
    if tensor.grad_fn is None:
        return False
    if tensor.dtype == torch.float32:
        return False
    if tensor.numel() == 0 or tensor.numel() % 64 != 0 or tensor.numel() < 32768:
        return False
    if tensor.storage().size() == 0 or tensor.storage_offset() != 0 \
        or tensor.storage().size() != tensor.numel():
        return False
    if tensor.grad_fn and "CheckpointWithout" in str(tensor.grad_fn):
        return False
    return True
