# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import List
from collections import namedtuple

from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.module.communication.communication_profile import ProfileTimeInfo
from mindspeed.auto_settings.utils.logger import get_logger

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
    def get_communication_info_from_profile(self, hcom_info_tage_id):
        pass

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
        """Trigger all the linear models to fit.

        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, search_cfg: SearchConfig):
        """Predict the communication time based on the given model configuration searched.

        :param search_cfg: The configuration of the search.
        :return: The predicted communication time.
        """
        pass

    @abc.abstractmethod
    def debug(self, config_list: List[SearchConfig]):
        """Print model configurations and the linear models' samples and fitted parameters.

    :return:
        """
        pass
