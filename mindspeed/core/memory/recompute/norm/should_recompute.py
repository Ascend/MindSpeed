# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.core.memory.recompute.recompute_common import should_recompute


def should_recompute_norm(layer_number, config):
    if not config.recompute_norm or layer_number is None:
        return False
    return should_recompute(config, layer_number, config.recompute_norm_num_layers)
