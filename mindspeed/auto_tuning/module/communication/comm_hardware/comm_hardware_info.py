import json
import os
import sys
from enum import Enum
from pathlib import Path
from megatron.training import get_args
from mindspeed.auto_tuning.utils.file_utils import check_file_size
from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import MindSpeedSettings as Settings


class HccsDev(Enum):
    hccs_dev_num_910b = 8
    # A3
    hccs_dev_num_910_9391 = 64
    hccs_dev_num_910_93 = 384
    hccs_dev_num_910_93_roce = 16
    # A5
    hccs_dev_num_910_95 = 64


_A5_HCCS_FILENAME = "A5_hccs_info.json"
_A5_ROCE_FILENAME = "Topo.json" # 文件名
_RANKTABLE_FILENAME = "ranktable_info.json"


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
        if "910_95" in device_type:
            self.a5_hccs = HccsA5Top()
            self.a5_roce = RoceA5Top()
            self.max_hccs_rank_num = HccsDev.hccs_dev_num_910_95.value
        if "910B" in device_type:
            self.max_hccs_rank_num = HccsDev.hccs_dev_num_910b.value
        with Path(__file__).parent.joinpath(_RANKTABLE_FILENAME).open(encoding="utf-8") as f:
            check_file_size(f)
            tanktable_dict = json.load(f)
            for item in tanktable_dict["dict"]:
                if item["device_num"] <= 0 or not isinstance(item["device_num"], int):
                    print("device_num Illegal:", item["device_num"])
                    self.max_hccs_rank_num = 0
                    return
                min(self.max_hccs_rank_num, item["device_num"])

    def calbandwidth(self, bandwidth_910b, min_domain):
        # roce
        if min_domain > self.max_hccs_rank_num:
            if "910_95" in self.hard_type:
                args = get_args()
                return self.a5_roce.calbandwidth(str(args.master_addr))
            return 1
        # hccs
        if "910B" in self.hard_type:
            return bandwidth_910b
        if "910_93" in self.hard_type:
            return 1
        if "910_95" in self.hard_type:
            self.a5_hccs.calbandwidth([4])
            return self.a5_hccs.bandwidth
        return 1


class HccsA5Top(object):
    def __init__(self):
        self.hccs_a5_bandwidth = [[]]
        self.x_max = 0
        self.y_max = 0
        self.x_y_max = 0
        self.logger = get_logger("CommPerfPredictor")

        with Path(__file__).parent.joinpath(_A5_HCCS_FILENAME).open(encoding="utf-8") as f:
            check_file_size(f)
            a5_hccs_dict = json.load(f)
            # "dict" x,y,bandwidth
            self.hccs_a5_bandwidth = a5_hccs_dict["dict"]
        self.bandwidth = 0

    def calbandwidth(self, top):
        if len(top) == 1:
            input_dev_num = top[0]
            x = 1
            y = 1
            if input_dev_num > 8:
                x = 8
                y = int(input_dev_num / 8)
            else:
                x = input_dev_num
        if len(top) == 2:
            x = top[0]
            y = top[1]
        self.bandwidth = self.calbandwidthxy(x, y)
        if self.bandwidth == 0:
            self.logger.error("A5 hccs dict has not info with x = ", x, "y = ", y)

    def calbandwidthxy(self, x, y):
        for item in self.hccs_a5_bandwidth:
            if item[0] == x and item[1] == y:
                self.bandwidth = item[2]
                return item[2]
        for item in self.hccs_a5_bandwidth:
            if item[0] * item[1] == x * y:
                return item[y]
        return 0


class RoceA5Top(object):
    def __init__(self):
        self.a5_roce_dict = None
        self.bandwidth = 0
        self.logger = get_logger("CommPerfPredictor")
        self._validate_and_load_topo()
                
    def calbandwidth(self, master_ip):
        leaf_bandwidths = None
        local_bandwidth = None
        if not self.a5_roce_dict:
            return 1
        try:
            for switches in self.a5_roce_dict["switches"]:
                if switches["ip"] in master_ip:
                    for elem in switches["npu_info_list"]:
                        if local_bandwidth is None or local_bandwidth > elem["port_bandwidths"]:
                            local_bandwidth = elem["port_bandwidths"]

                if len(switches["npu_info_list"]) > 0:
                    for elem in switches["npu_info_list"]:
                        if leaf_bandwidths is None or leaf_bandwidths > elem["port_bandwidths"]:
                            leaf_bandwidths = elem["port_bandwidths"]
        except KeyError as e:
            raise AssertionError(
                    "double check your topo file, the key is wrong") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during communication performance predictor:{str(e)}"
            ) from e
        
        if local_bandwidth is None:
            local_bandwidth = 400000 # default
        self.bandwidth = leaf_bandwidths / local_bandwidth
        return self.bandwidth 

    def _validate_and_load_topo(self):
        topo_path = Path(Settings().work_dir) / _A5_ROCE_FILENAME
        if not topo_path.is_file():
            self.logger.debug("No Topo File or Wrong Dictionary")
            return
        try:
            with topo_path.open(encoding="utf-8") as f:
                check_file_size(f)
                self.a5_roce_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(
                "Topo json load failed",
                "The 910_95 needs to have topo file for communication model") from e
        self._validate_json_structure(self.a5_roce_dict)

    def _validate_json_structure(self, data):
        if "switches" not in data:
            raise AssertionError("double check your topo file, the key is wrong")
        if len(data["switches"]) == 0:
            raise AssertionError("double check your topo file, there is no switches")
        for switch in data["switches"]:
            if "npu_info_list" not in switch:
                raise AssertionError("double check your topo file, the key is wrong")
            if len(switch["npu_info_list"]) == 0:
                continue  # spine switches
            for npu_info in switch["npu_info_list"]:
                if "port_bandwidths" not in npu_info:
                    raise AssertionError("double check your topo file, the key is wrong")
                try:
                    npu_info["port_bandwidths"] = float(npu_info["port_bandwidths"])
                except Exception as e:
                    raise AssertionError("port_bandwidths load fail") from e