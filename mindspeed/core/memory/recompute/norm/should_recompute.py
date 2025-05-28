# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.core.memory.recompute.recompute_common import should_recompute
from mindspeed.core.pipeline_parallel.ripipe_schedules import get_pipeline_checkpoint_manager


def should_recompute_norm(layer_number, config):
    if not config.recompute_norm or layer_number is None:
        return False

    if config.recompute_in_bubble or config.recompute_in_advance:
        pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(config.virtual_pipeline_model_parallel_size)
        if pipeline_checkpoint_manager.chunk_do_recompute:
            return False
        elif config.recompute_in_bubble:
            return True
    
    return should_recompute(config, layer_number, config.recompute_norm_num_layers)
