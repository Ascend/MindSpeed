import os
from enum import Enum

from mindspeed.auto_settings.config.system_config import get_system_config


class HccsDev(Enum):
    hccs_dev_num_910b = 8
    # A3
    hccs_dev_num_910_9391 = 64
    hccs_dev_num_910_93 = 384
    hccs_dev_num_910_93_roce = 16
    # A5
    hccs_dev_num_910_94 = 64


class CommHardInfo(object):
    def __init__(self, device_type):
        self.max_hccs_rank_num = 8
        self.hard_type = device_type
        if "910_93" in device_type:
            self.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93.value
        if os.getenv("HCCL_INTER_HCCS_DISABLE", None):
            self.max_hccs_rank_num = HccsDev.hccs_dev_num_910_93_roce.value
        if "910_9391" in device_type:
            self.max_hccs_rank_num = HccsDev.hccs_dev_num_910_9391.value
        if "910B" in device_type:
            self.max_hccs_rank_num = HccsDev.hccs_dev_num_910b.value


    def calbandwidth(self, bandwidth_910b, min_domain):
        # roce
        system_config = get_system_config()
        if system_config.nnodes > 1 and min_domain > self.max_hccs_rank_num:
            return 1
        # hccs
        if "910B" in self.hard_type:
            return bandwidth_910b
        if "910_93" in self.hard_type:
            return 1
        return 1

