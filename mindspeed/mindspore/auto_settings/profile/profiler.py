import os
import time
from typing import Any, Dict, List, Optional
import torch.distributed as dist
from mindspeed.auto_settings.config.search_config import SearchConfig, ExecutorFlag
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_settings.utils.utils import check_file_exists, get_prof_dir


def profiler_run(self, output_filename: str,
        cfg: Optional[SearchConfig] = None,
        flag: ExecutorFlag = ExecutorFlag.PROFILE):
    """
    running on master node
    """
    self.init()
    if flag == ExecutorFlag.PARSE_ARGS:
        return_code = self._prepare(output_filename, cfg=cfg, flag=flag)
        return return_code
    dist.barrier()
    dist.broadcast_object_list([output_filename, cfg, flag])
    return_code = self._prepare(output_filename, cfg=cfg, flag=flag)
    dist.barrier()
    return return_code


def profile(self, configs):
    """
    get profiling data
    """
    profile_results = []
    self._logger.info("<==========Begin to profile==========>")
    for idx, (config, file_name) in enumerate(configs):
        time.sleep(10)
        os.system(f'export HCCL_IF_BASE_PORT={9671 + idx}')
        if not check_file_exists(file_name):
            self._logger.info('<==========the %s/%s loop==========>', str(idx), str(len(configs)))
            self._logger.info("profile_db_configs (tp, pp, dp, cp, ep, #layers, seq_len):")
            self.run(file_name, config, flag=config.profile_type)
        if config.profile_type == ExecutorFlag.PROFILE:
            file_path = os.path.join(get_system_config().work_dir, get_prof_dir(config))
            profiling_node_parse = GatherNodeProfiling(file_path)
            profiling_res = profiling_node_parse.fuse_node_pkl()
            profile_results.append([config, profiling_res])
    self._logger.info("<==========Finished profiling==========>")
    return profile_results
