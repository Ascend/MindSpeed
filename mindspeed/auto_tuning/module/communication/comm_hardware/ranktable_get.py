# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import json
from kubernetes import client, config
from mindspeed.auto_tuning.utils.logger import get_logger

config.kube_config.load_kube_config(config_file="/root/.kube/config")
 
#获取API的CoreV1Api版本对象
v1 = client.CoreV1Api()
 
config_maps = v1.list_namespaced_config_map('default')

logger = get_logger("ranktable_get")


class CommRankrTable(object):
    """CommRankrTable"""
 
    def __init__(self, server_list=None):
        self.device_num = 0
        self.hyperplane_list = []
        self._get_rankrable_form_server_list(server_list)
 
    def _get_rankrable_form_server_list(self, server_list):
        for server in server_list:
            hyper_plane = HyperPlaneInfo(server)
            self.device_num += hyper_plane.device_num
            self.hyperplane_list.append(hyper_plane)


class HyperPlaneInfo(object):
    """HyperPlaneInfo"""
 
    def __init__(self, server=None):
        self.server_id = server["server_id"]
        self.server_name = server['server_name']
        self.device_num = 0
        self.device_ip_list = []
        self._get_info_form_server(server)
 
    def _get_info_form_server(self, server):
        for device in server['device']:
            self.device_num += 1
            self.device_ip_list.append(device['device_ip'])
 
 
# 得到的config_map是一个list,需要层层剥离以判断是否是目标
def _get_hccl_class_from_config_maps(maps, hccl_class_list):
    for config_map in maps.items:
        if 'data' in dir(config_map):
            if isinstance(config_map.data, dict):
                if 'hccl.json' in config_map.data:
                    hccl_json = (config_map.data['hccl.json'])
                    hccl_class = json.loads(hccl_json)
                    hccl_class_list.append(hccl_class)
                    
 
def _get_server_list_from_hccl_class(hccl_class_list, job_rank_table_list):
    for hccl_class in hccl_class_list:
        if 'server_list' in hccl_class:
            comm_ranbk_tabnle = CommRankrTable(hccl_class['server_list'])
            job_rank_table_list.append(comm_ranbk_tabnle)
 
_RANKTABLE_FILENAME = "ranktable_info.json"
 
if	__name__ ==	"__main__":
    tanktable_dict = {"dict": []}
    hccl_class_list = []
    job_rank_table_list = []
    _get_hccl_class_from_config_maps(config_maps, hccl_class_list)
    _get_server_list_from_hccl_class(hccl_class_list, job_rank_table_list)
    logger.info("=================")
    if len(job_rank_table_list) > 0:
        logger.info("total device:", job_rank_table_list[0].device_num)
        logger.info("Hyperplane num:", len(job_rank_table_list[0].hyperplane_list))
        for index, hyper_plane in enumerate(job_rank_table_list[0].hyperplane_list):
            logger.info("     index:", index, " total device:", hyper_plane.device_num)
            tanktable_dict['dict'].append({"index": index, "device_num": hyper_plane.device_num})
        with open(_RANKTABLE_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(tanktable_dict, f, ensure_ascii=False)
    else:
        logger.info("no job in k8s")
