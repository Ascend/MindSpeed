# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from functools import wraps

import torch
import torch_npu

from megatron.training import get_args
from megatron.training.utils import print_rank_0

from mindspeed.core.hccl_buffer.hccl_adaptive_func import parse_hccl_op_mode_string, _HCCL_OP_MODE


def get_nccl_options_wrapper(get_nccl_options):
    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        args = get_args()
        options = get_nccl_options(pg_name, nccl_comm_cfgs)
        if args.hccl_op_mode and _HCCL_OP_MODE.get(pg_name) is not None:
            original_hccl_config = options.hccl_config
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            original_hccl_config["hccl_op_expansion_mode"] = _HCCL_OP_MODE[pg_name]
            options.hccl_config = original_hccl_config
        return options

    return wrapper


def hccl_op_mode_set_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(*args, **kwargs):
        config = get_args()
        if config.hccl_op_mode is not None:
            parse_hccl_op_mode_string(config.hccl_op_mode)
            print_rank_0(f"hccl_op_mode_set: {_HCCL_OP_MODE}")

        return initialize_model_parallel(*args, **kwargs)
    return wrapper
