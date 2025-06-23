#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from mindspeed.auto_settings.utils.utils import get_module_info, get_black_prof_file
from mindspeed.auto_settings.config.search_config import SearchConfig


def get_module_time(self, config: SearchConfig, module_name, node_rank):
    tmp_config = self.crop_config(config)
    path = get_black_prof_file(tmp_config)
    module = get_module_info(path, module_name)
    fwd_time = float(module.get('time', float('inf'))) * 1000
    forward_step_time = get_module_info(path, 'forward_step_time') * 1000
    backward_step_time = get_module_info(path, 'backward_step_time') * 1000
    return fwd_time, fwd_time / forward_step_time * backward_step_time
