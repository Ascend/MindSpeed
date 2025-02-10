# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
from megatron.training import get_args
from .config.dist_train_config import get_all_config


def initialize_distributed_wrapper(_initialize_distributed):
    @wraps(_initialize_distributed)
    def wrapper():
        args = get_args()
        temp_world_size = args.world_size
        true_world_size = 0
        for cfg in get_all_config().values():
            true_world_size += cfg.world_size
        args.world_size = true_world_size

        _initialize_distributed()

        args.world_size = temp_world_size

    return wrapper
