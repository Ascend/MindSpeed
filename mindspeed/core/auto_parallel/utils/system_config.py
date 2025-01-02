# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
from dataclasses import dataclass, field

import torch


@dataclass
class SystemConfig:

    nnodes: int

    nproc_per_node: int

    node_rank: int

    master_addr: str

    master_port: int

    target_nnodes: int

    search_algo: str

    work_dir: str

    log_level: int

    # 支持搜索的并行维度个数
    search_dimensions: int

    # 实际用于搜索的小集群规模
    world_size: int = field(init=0)

    # 需要搜索的大集群规模
    target_world_size: int = field(init=0)

    max_available_memory: float = field(init=0.)

    def __post_init__(self):
        self.world_size = self.nnodes * self.nproc_per_node
        self.target_world_size = self.target_nnodes * self.nproc_per_node
        self.max_available_memory = torch.npu.get_device_properties(0).total_memory / (1024 ** 3)
    
    @property
    def cache_path(self):
        work_dir = self.work_dir
        if not self.work_dir.endswith(os.sep):
            work_dir += os.sep
        
        try:
            os.makedirs(work_dir, exist_ok=True)
        except BaseException:
            work_dir = os.getcwd()
        
        return work_dir


_SYSTEM_CONFIG: SystemConfig = None


def set_system_config(sys_config):
    global _SYSTEM_CONFIG
    if _SYSTEM_CONFIG is not None:
        raise AssertionError('SYSTEM_CONFIG has been initialized')
    _SYSTEM_CONFIG = sys_config


def get_system_config():
    global _SYSTEM_CONFIG
    if _SYSTEM_CONFIG is None:
        raise AssertionError('SYSTEM_CONFIG is not initialized')
    return _SYSTEM_CONFIG
