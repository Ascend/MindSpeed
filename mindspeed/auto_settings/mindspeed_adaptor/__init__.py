# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import List, Tuple
from argparse import Namespace
import os
import re
import stat
import json

import torch.cuda as cuda
import torch.distributed as dist
from torch.nn import Module

from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.utils.file_utils import check_file_size, restricted_write
from mindspeed.auto_settings.utils.mem_utils import mem_b_to_mb


import sys
import signal
import faulthandler

faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)


def get_settings(args: Namespace, filename: str) -> PostInfo:
    open_flags = os.O_RDONLY
    file_mode = stat.S_IWUSR | stat.S_IRUSR
    open_mode = "r"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pkl = PostInfo()

    pkl.model_config = ModelConfig()
    for k in vars(pkl.model_config).keys():
        if not getattr(pkl.model_config, k, True):
            setattr(pkl.model_config, k, getattr(args, k, None))
    model_name = filename.split('/')[-2]
    if 'gpt' in model_name:
        pkl.model_config.num_layers = 32
    elif 'vit' in model_name:
        pkl.model_config.num_layers = 24
    pkl.model_config.world_size = world_size
    pkl.devices_per_node = cuda.device_count()
    pkl.nnodes = world_size // pkl.devices_per_node
    pkl.node_rank = rank // pkl.devices_per_node
    pkl.device_type = cuda.get_device_name()
    pkl.wait_timeout = (int(os.getenv("HCCL_EXEC_TIMEOUT", "1836")) // 68) * 68

    group = dist.new_group(backend=dist.Backend.GLOO)
    local_mem_cap, _ = cuda.mem_get_info(device=rank % pkl.devices_per_node)
    mem_caps = [0] * world_size
    dist.all_gather_object(mem_caps, local_mem_cap, group=group)
    pkl.memory_cap = mem_b_to_mb(min(mem_caps))
    dist.barrier(group=group)
    dist.destroy_process_group(group=group)

    driver_version_path = os.path.join(os.sep, "usr", "local", "Ascend", "driver", "version.info")
    with os.fdopen(os.open(driver_version_path, open_flags, mode=file_mode), mode=open_mode) as file:
        check_file_size(file)
        lines = filter(lambda line: not line.startswith("#"), file.readlines())
        content = "\n".join(lines)
        driver_version = re.findall(r"package_version=(\S+)", content)
        pkl.driver_version = driver_version[0] if driver_version else "N/A"

    cann_path = os.getenv("ASCEND_HOME_PATH", os.path.join( \
        os.sep, "usr", "local", "Ascend", "ascend-toolkit", "latest"))
    cann_version_path = os.path.join(cann_path, "version.cfg")
    with os.fdopen(os.open(cann_version_path, open_flags, mode=file_mode), mode=open_mode) as file:
        check_file_size(file)
        lines = filter(lambda line: not line.startswith("#"), file.readlines())
        content = "\n".join(lines)
        cann_version = re.findall(r"toolkit_installed_version=\[([^:]+):", content)
        pkl.cann_version = cann_version[0] if cann_version else "N/A"

    if rank % pkl.devices_per_node == 0:
        restricted_write(filename, pkl)

    return pkl



def get_model_params(
    model: List[Module],
    pipeline_model_parallel_rank: int,
    output_path: str,
    mm_data: str = None
) -> List[Tuple[str, int]]:
    model_params: List[Tuple[str, int]] = list()

    def traverse_module_layers(module: Module, prefix: str):
        new_prefix = f"{prefix}{module.__class__.__name__}."

        if not list(module.children()):
            for param_name, param in module.named_parameters():
                model_params.append((f"{new_prefix}{param_name}", param.numel()))
            return

        for sub_module in module.children():
            traverse_module_layers(sub_module, new_prefix)

    for module in model:
        traverse_module_layers(module, str())

    if "auto_settings_static_model" in output_path:
        total_model_params = [None] * dist.get_world_size()
        dist.all_gather_object(total_model_params, (pipeline_model_parallel_rank, model_params), group=dist.group.WORLD)
    else:
        total_model_params = [None] * dist.get_world_size()
        group = dist.new_group([0], backend=dist.Backend.GLOO)
        dist.all_gather_object(total_model_params, (pipeline_model_parallel_rank, model_params), group=group)
    if dist.get_rank() % cuda.device_count() == 0:
        restricted_write(output_path, total_model_params)

    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group(group=dist.group.WORLD)

    return model_params

