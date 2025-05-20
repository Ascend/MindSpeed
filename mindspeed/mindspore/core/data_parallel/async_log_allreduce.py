# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import mindspore

from megatron.core import mpu


def get_async_reduced_loss_value(x, key):
    """
    Retrieves the reduced loss value after waiting for the completion of the async all-reduce operation.

    Args:
    x (tuple): A tuple containing two elements:
               - A dictionary where the key corresponds to the loss value.
               - A `torch.distributed.Work` object used for waiting for the completion of the all-reduce operation.
    key (str): The key used to access the loss value from the dictionary.

    Returns:
    val: The loss value retrieved from the dictionary using the specified key. The type of `val` depends on the type stored in the dictionary for the given key.

    Raises:
    AssertionError: If the second element of `x` is not of type `torch.distributed.Work`, an assertion error is raised.
    """
    # Wait until the loss allreduce execution is complete.
    # In most cases, the loss allreduce has already completed when the program execution reaches this point.

    val = x[0][key]
    handle = x[1]
    if not isinstance(handle, mindspore._c_expression.CommHandle):
        raise AssertionError(
            f"when using --async-log-allreduce , type of the first input must be {mindspore._c_expression.CommHandle}, but got {type(handle)}.")
    handle.wait()

    val = val / torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    return val
