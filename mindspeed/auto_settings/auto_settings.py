"""
自动并行配置搜索主要入口
"""
import os

import torch
import logging

from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.config.model_config import set_model_config, ModelConfig, get_model_config, \
    update_model_config
from mindspeed.auto_settings.config.search_config import ExecutorFlag
from mindspeed.auto_settings.config.system_config import set_system_config, get_system_config
from mindspeed.auto_settings.profile.profiler import Profiler
from mindspeed.auto_settings.module.searcher import WhiteSearcher, BlackSearcher, MixedSearcher
from mindspeed.auto_settings.search_space import SearchSpace
from mindspeed.auto_settings.utils.file_utils import restricted_read
from mindspeed.auto_settings.utils.logger import get_logger


class AutoSettings(object):

    def __init__(self):
        # self.patcher = Patcher()
        self.search_spaces = SearchSpace()
        self.profiler = Profiler()
        self.logger = get_logger("AutoSettings")

    def init(self, args):
        """
        初始化相关配置
        """
        self._init_configs(args)
        self._init_hardware(args)
        self._init_global_group()

    def _init_hardware(self, args):
        self.profiler.run(PostInfo.FILENAME, None, ExecutorFlag.PARSE_ARGS)
        post_info = restricted_read(os.path.join(str(get_system_config().work_dir), PostInfo.FILENAME))
        get_system_config().load_settings(post_info)
        update_model_config(post_info.model_config)

    def _init_configs(self, args):
        """
        初始化相关配置
        """
        set_system_config(args)
        set_model_config(args)

    def _init_global_group(self):
        """
        初始化相关
        """
        sys_config = get_system_config()
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.GLOO,
            rank=sys_config.node_rank,
            world_size=sys_config.nnodes
        )

    def _get_searcher(self, search_type):
        """
        根据用户配置获取对应的搜索器
        """
        if search_type == "white":
            return WhiteSearcher()
        if search_type == "black":
            return BlackSearcher()
        if search_type == "mixed":
            searcher = MixedSearcher()
            searcher.set_white_topk(5)
            return searcher
        return MixedSearcher()

    def search(self, args):
        """
        搜索入口
        """
        self.logger.info("model config is that:\n%s", str(get_model_config()))
        search_type = args.auto_settings_type
        searcher = self._get_searcher(search_type)
        # 搜索空间
        search_configs = self.search_spaces.build_search_spaces()
        final_configs = searcher.search(configs=search_configs, topk=3)

        return final_configs

    def auto_settings(self, args):
        """
        入口函数
        """
        self.init(args)
        if get_system_config().node_rank != 0:
            self.profiler.run_on_slaves(args)
            return
        final_configs = self.search(args)
        self.logger.info("<==========Final config generated==========>")
        self.logger.info("The recommended configurations are:")
        for i, final_cfg in enumerate(final_configs):
            if final_cfg:
                self.logger.info("<==========Top #%s config==========>", str(i))
                if self.logger.getEffectiveLevel() == logging.DEBUG:
                    self.logger.debug("\n%s", str(final_cfg))
                else:
                    self.logger.info("\n%s", str(final_cfg))
        self.logger.info("<==========Start training==========>")
        return final_configs
