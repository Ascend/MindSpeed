# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import time
from enum import Enum

from .swap_policy_config import swap_policy_config


class PrintLevel(Enum):
    DEBUG = 0
    INFO = 1
    NONE = 2


def print_with_rank(message, prefix="", print_level=PrintLevel.DEBUG):
    if swap_policy_config.print_level > print_level.value:
        return

    rank = swap_policy_config.rank
    print_rank = swap_policy_config.print_rank
    if print_rank == -1:
        print(f"[{print_level.name}] rank[{rank}] [{prefix}]: {message}", flush=True)
    else:
        if rank == print_rank:
            print(f"[{print_level.name}] rank[{rank}] [{prefix}]: {message}", flush=True)


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print_with_rank(
            f"Function {func.__name__} takes {end_time - start_time} seconds to execute.",
            prefix="timer",
            print_level=PrintLevel.INFO,
        )
        return result

    return wrapper
