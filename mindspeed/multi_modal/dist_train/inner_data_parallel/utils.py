# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from megatron.core.parallel_state import get_data_parallel_world_size
from ..config.dist_train_config import get_all_config
from ..parallel_state import is_in_subworld


def need_inner_data_parallel():
    # only support for vit model
    if not is_in_subworld("vit"):
        return False
    if not any(cfg.main_dp for cfg in get_all_config().values()):
        return False
    return get_data_parallel_world_size() // get_global_data_parallel_size() > 1


def get_global_data_parallel_size():
    """Return main data parallel size"""
    all_config = get_all_config()
    global_dp_size = 0
    for config in all_config.values():
        if config.main_dp:
            global_dp_size = config.world_size // \
                (config.tensor_model_parallel_size * config.pipeline_model_parallel_size * config.context_parallel_size)
    if global_dp_size == 0:
        raise AssertionError("No Main DP")
    return global_dp_size
