# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Callable, Any
from collections import defaultdict
import warnings

from torch import Tensor

cached_weight_pool = defaultdict(lambda: {})
num_quantized = 0
num_quant_op_reduced = 0
current_weight = None
print_statistics = True
optimizer_hooked = False


def set_current_cacheable_weight(obj):
    global current_weight
    current_weight = obj


def cached_quant(
        x: Tensor,
        quantizer: Callable[[Tensor], Any],
        key: str = None,
        **kwargs,
):
    global num_quantized, num_quant_op_reduced, optimizer_hooked
    if not optimizer_hooked:
        warnings.warn(
            "optimizer.step is not hooked, quantizer will be called without caching, which will impair the training performance")
        return quantizer(x, **kwargs)

    key = id(current_weight) if key is None else key

    if key in cached_weight_pool:
        num_quant_op_reduced += 1
        return cached_weight_pool[key]

    result = quantizer(x, **kwargs)

    cached_weight_pool[key] = result
    num_quantized += 1

    return result


def reset_cache_and_weight(model, optimizer):
    global num_quantized, num_quant_op_reduced, print_statistics

    for _, data in cached_weight_pool.items():
        for di in data:
            if isinstance(di, Tensor):
                di.untyped_storage().resize_(0)
    cached_weight_pool.clear()

    if print_statistics:
        print(f"fp8_cache_quantized_weight: num_quantized={num_quantized}, num_quant_op_reduced={num_quant_op_reduced}",
              flush=True)
        print_statistics = False

    num_quantized = 0
    num_quant_op_reduced = 0


def hook_optimizer_step(model, optimizer):
    global optimizer_hooked
    if optimizer_hooked:
        return

    optimizer_step = optimizer.step

    def cached_optimizer_step():
        reset_cache_and_weight(model, optimizer)
        return optimizer_step()

    optimizer.step = cached_optimizer_step
    optimizer_hooked = True
