from typing import List, Tuple
from argparse import Namespace
import os
import re
import stat

import torch.cuda as cuda
import torch.distributed as dist
from torch.nn import Module

from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.config.model_config import ModelConfig, set_model_config, get_model_config
from mindspeed.auto_settings.utils.file_utils import check_file_size, restricted_write
from mindspeed.auto_settings.utils.mem_utils import mem_b_to_mb


def get_settings(args: Namespace, filename: str) -> PostInfo:
    open_flags = os.O_RDONLY
    file_mode = stat.S_IWUSR | stat.S_IRUSR
    open_mode = "r"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pkl = PostInfo()

    set_model_config(args)
    pkl.model_config = get_model_config()

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
        lines = filter(lambda l: not l.startswith("#"), file.readlines())
        content = "\n".join(lines)
        driver_version = re.findall(r"package_version=(\S+)", content)
        pkl.driver_version = driver_version[0] if driver_version else "N/A"

    cann_path = os.getenv("ASCEND_HOME_PATH",
                          os.path.join(os.sep, "usr", "local", "Ascend", "ascend-toolkit", "latest"))
    cann_version_path = os.path.join(cann_path, "version.cfg")
    with os.fdopen(os.open(cann_version_path, open_flags, mode=file_mode), mode=open_mode) as file:
        check_file_size(file)
        lines = filter(lambda l: not l.startswith("#"), file.readlines())
        content = "\n".join(lines)
        cann_version = re.findall(r"toolkit_installed_version=\[([^:]+):", content)
        pkl.cann_version = cann_version[0] if cann_version else "N/A"

    if rank % pkl.devices_per_node == 0:
        restricted_write(filename, pkl)

    return pkl


def get_model_params(
        model: List[Module],
        pipeline_model_parallel_rank: int,
        output_path: str
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

    total_model_params = [None] * dist.get_world_size()
    group = dist.new_group(backend=dist.Backend.GLOO)
    dist.all_gather_object(total_model_params, (pipeline_model_parallel_rank, model_params), group=group)
    if dist.get_rank() % cuda.device_count() == 0:
        restricted_write(output_path, total_model_params)

    dist.barrier(group=group)
    dist.destroy_process_group(group=group)

    return model_params
