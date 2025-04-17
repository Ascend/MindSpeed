import os
from enum import IntEnum

import torch.distributed as dist
from typing import Any, Dict, List, Optional

from mindspeed.auto_settings.config.search_config import SearchConfig, ExecutorFlag
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.parse.profiling_parse.profiling_node_parse import GatherNodeProfiling
from mindspeed.auto_settings.profile.argv import BaseArgv, FilterArgv, SearchConfigArgv
from mindspeed.auto_settings.profile.runner import Runner
from mindspeed.auto_settings.utils.file_utils import restricted_write
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.singleton import Singleton
from mindspeed.auto_settings.utils.utils import check_file_exists, get_prof_dir


class Profiler(metaclass=Singleton):
    PARSE_ARGS_ENV = "OOTB_OPTIMIZER_PARSE_ARGS"
    PARSE_MODEL_ENV = "OOTB_OPTIMIZER_PARSE_MODEL"
    PROFILING_ENV = "OOTB_OPTIMIZER_PROFILING"
    PROFILING_ENV_BLACK = "OOTB_OPTIMIZER_PROFILING_BLACK"
    ENABLED_ENV_MARKER = "TRUE"

    def __init__(self):
        self._logger = get_logger("Profiler")

    def init(self):
        self.runner = Runner()

    def run(self, output_filename: str,
                      cfg: Optional[SearchConfig] = None,
                      flag: ExecutorFlag = ExecutorFlag.PROFILE):
        """
        运行在主节点
        """
        self.init()
        if flag == ExecutorFlag.PARSE_ARGS:
            return_code = self._prepare(output_filename, cfg=cfg, flag=flag)
            return return_code
        dist.monitored_barrier(wait_all_ranks=True)
        dist.broadcast_object_list([output_filename, cfg, flag])
        return_code = self._prepare(output_filename, cfg=cfg, flag=flag)
        dist.barrier()
        return return_code

    def run_on_slaves(self, args):
        """
        运行在从节点上
        """
        self.init()
        count = 0
        while True:
            try:
                self._logger.info(f"[#{count}] Waiting for master.....")
                dist.monitored_barrier(wait_all_ranks=True)
                bcast_list: List[Any] = [None] * 3
                dist.broadcast_object_list(bcast_list)
                output_filename, cfg, flag = bcast_list
                self._prepare(output_filename, cfg=cfg, flag=flag)
                dist.barrier()
            except RuntimeError as e:
                if "successfully reached monitoredBarrier" in str(e):
                    count += 1
                    if count > 10:
                        self._logger.critical(f"Wait timeout, shutting down.")
                        raise e
                elif "Connection closed by peer" in str(e):
                    self._logger.info("Master shuts down, exiting.....")
                    return
                else:
                    raise e

    def _prepare(
            self,
            output_filename: str,
            cfg: Optional[SearchConfig] = None,
            flag: ExecutorFlag = ExecutorFlag.PROFILE
    ) -> int:
        system_config = get_system_config()
        work_dir = system_config.work_dir
        save_path = os.path.join(work_dir, output_filename)
        modified_env = self._update_env(flag)
        modified_argv = self._update_argv(save_path, cfg)
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        if cfg:
            restricted_write(os.path.join(system_config.work_dir, f"at_{system_config.node_rank}.pkl"), cfg)
        return_code = self.runner.run(modified_argv, modified_env)

        return return_code

    def _update_env(self, flag: ExecutorFlag) -> Dict[str, str]:
        """
        更新环境变量
        """
        env = self.runner.get_base_env()

        env.pop(self.PARSE_ARGS_ENV, None)
        env.pop(self.PARSE_MODEL_ENV, None)
        env.pop(self.PROFILING_ENV, None)
        env.pop(self.PROFILING_ENV_BLACK, None)
        if flag == ExecutorFlag.PARSE_ARGS:
            env.update({self.PARSE_ARGS_ENV: self.ENABLED_ENV_MARKER})
        elif flag == ExecutorFlag.PARSE_MODEL:
            env.update({self.PARSE_MODEL_ENV: self.ENABLED_ENV_MARKER})
        elif flag == ExecutorFlag.PROFILE:
            env.update({self.PROFILING_ENV: self.ENABLED_ENV_MARKER})
        elif flag == ExecutorFlag.PROFILE_BLACK:
            env.update({self.PROFILING_ENV_BLACK: self.ENABLED_ENV_MARKER})
        return env

    def _update_argv(
            self,
            save_path: str,
            cfg: Optional[SearchConfig]
    ) -> List[str]:
        """
        更新运行参数
        """
        argv = self.runner.get_base_argv()
        BaseArgv.base_argv(argv, save_path)
        FilterArgv.filter_argv(argv)
        SearchConfigArgv.update_argv(argv, config=cfg)
        return argv

    def profile(self, configs):
        """
        对相关数据并行进行数据采集
        """
        profile_results = []
        self._logger.info("<==========Begin to profile==========>")
        for idx, (config, file_name) in enumerate(configs):
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
