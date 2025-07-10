# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import abc
from typing import List
from collections import namedtuple

from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.communication.communication_profile import ProfileTimeInfo
from mindspeed.auto_tuning.utils.logger import get_logger

SimpleParallelCfg = namedtuple(
    "SimpleParallelCfg", field_names=["config_no", "tp", "cp", "dp", "ep", "pp", "vp"]
)


class CommPerfPredictor:
    def __init__(self, hard_info):
        self.logger = get_logger("CommPerfPredictor")
        self.max_hccs_rank_num = hard_info.max_hccs_rank_num
        self.hard_info = hard_info
        self.debug_info_list = []

    @abc.abstractmethod
    def receive_samples_from_profiling(
        self, config_no, model_config: SearchConfig, profile_info: ProfileTimeInfo
    ):
        """Parse profiling info and extract the samples including 'x'(s) and 'y' and add to the
        linear models.

        :param model_config:
        :param profile_info:
        :return:
        """
        pass

    @abc.abstractmethod
    def fit(self):
        """trigger all the linear models to fit.

        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, search_cfg: SearchConfig):
        """Predict communication time according given  model config searched

        :param search_cfg:
        :return:
        """
        pass

    @abc.abstractmethod
    def debug(self, config_list: List[SearchConfig]):
        """Print model configs and the linear models' samples and fitted parameters.

        :return:
        """
        pass
