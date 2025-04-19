# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps

import torch
import torch_npu

from megatron.training import get_args
from megatron.training.utils import print_rank_0

from mindspeed.core.hccl_buffer.hccl_adaptive_func import hccl_buffer_auto_adaptive, _HCCL_GROUP_BUFFER


def get_nccl_options_wrapper(get_nccl_options):
    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        args = get_args()
        if args.hccl_group_buffer_adaptive:
            global _HCCL_GROUP_BUFFER
            if _HCCL_GROUP_BUFFER.get(pg_name) is not None:
                options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                options.hccl_config = {"hccl_buffer_size": _HCCL_GROUP_BUFFER[pg_name]}
                return options
        return get_nccl_options(pg_name, nccl_comm_cfgs)
    return wrapper


def hccl_buffer_adaptive_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(*args, **kwargs):
        config = get_args()
        global _HCCL_GROUP_BUFFER

        if config.hccl_group_buffer_adaptive:
            hccl_buffer_auto_adaptive(config)
            print_rank_0(f"hccl_group_buffer_adaptive: {_HCCL_GROUP_BUFFER}")

        return initialize_model_parallel(*args, **kwargs)
    return wrapper
