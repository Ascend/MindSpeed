# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
import json
import threading
import functools
import operator

import torch


class SingletonType(type):
    single_lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with SingletonType.single_lock:
            if not hasattr(cls, "_instance"):
                cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance
    

class GlobalMemoryBuffer:
    buffers_length = [0, 0, 0]
    buffers = [None, None, None]

    @staticmethod
    def get_tensor(shape: list, index):
        if index not in (0, 1, 2):
            raise AssertionError('index must be 0, 1, 2')

        data_type = torch.float16
        required_len = functools.reduce(operator.mul, shape, 1)

        if GlobalMemoryBuffer.buffers_length[index] < required_len:
            GlobalMemoryBuffer.buffers[index] = torch.empty(
                (required_len,), dtype=data_type, requires_grad=False, device=torch.cuda.current_device()
            )
            GlobalMemoryBuffer.buffers_length[index] = required_len
        return GlobalMemoryBuffer.buffers[index][0:required_len].view(*shape).uniform_()


class KVStore:
    kv_store = None

    @classmethod
    def init(cls):
        from .system_config import get_system_config

        sys_config = get_system_config()
        cls.kv_store = torch.distributed.TCPStore(
            host_name=sys_config.master_addr,
            port=sys_config.master_port + 2,
            world_size=sys_config.nnodes,
            is_master=sys_config.node_rank == 0
        )
    
    @classmethod
    def get(cls, key):
        if cls.kv_store is None:
            raise AssertionError("KVStore must be initialized")
        return cls.kv_store.get(key)
    
    @classmethod
    def set(cls, key, value):
        if cls.kv_store is None:
            raise AssertionError("KVStore must be initialized")
        cls.kv_store.set(key, value)
        

def get_cache_path():
    from . import logger
    from .model_config import get_model_config as get_config

    work_dir = get_config().args.work_dir
    if not work_dir.endswith(os.sep):
        work_dir += os.sep

    try:
        os.makedirs(work_dir, exist_ok=True)
    except BaseException:
        logger.warning(f"Create cache directory failed")
        work_dir = os.getcwd()

    return work_dir


def get_module_info(file_path, key, sub_key=None):
    from . import logger

    try:
        with open(file_path, 'r') as file:
            content = json.loads(file.read())
            if sub_key is None:
                return content[key]
            else:
                return content[key][sub_key]
    except FileNotFoundError:
        return float('inf')
    