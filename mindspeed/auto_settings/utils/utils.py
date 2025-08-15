import json
import os
import re
from typing import Optional

from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.search_config import SearchConfig, DISABLE_CP
from mindspeed.auto_settings.config.system_config import get_system_config


def check_file_exists(filename: str) -> bool:
    return os.path.exists(os.path.join(get_system_config().work_dir, filename))


def get_tp_for_profiling() -> int:
    tp = get_system_config().world_size // 4
    return min(tp, 4)


def get_num_warmup_micro_batches(config: SearchConfig, model_cfg: ModelConfig):
    """
    获取warmup micro_batches
    """
    if config.layers_per_vpp:
        num_model_chunks = config.num_layers // config.layers_per_vpp // config.pp
    else:
        num_model_chunks = 1
    pipeline_parallel_size = config.pp
    data_parallel_size = config.dp
    num_microbatches = model_cfg.gbs // (config.mbs * data_parallel_size)

    if pipeline_parallel_size <= 1:
        return 1, num_microbatches

    pipeline_parallel_size = pipeline_parallel_size
    pipeline_parallel_rank = 0
    total_num_micro_batches = num_microbatches * num_model_chunks
    if num_model_chunks == 1:
        num_warmup_micro_batches = pipeline_parallel_size - pipeline_parallel_rank - 1

    else:
        num_warmup_micro_batches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_micro_batches += (num_model_chunks - 1) * pipeline_parallel_size
    num_warmup_micro_batches += 1
    num_warmup_micro_batches = min(num_warmup_micro_batches, total_num_micro_batches)
    return num_warmup_micro_batches, num_microbatches


def get_seq_length_for_profiling(model_cfg: ModelConfig) -> int:
    if not DISABLE_CP:
        return max(model_cfg.seq_length, 8 * 1024)
    return min(model_cfg.seq_length, 32 * 1024)


def get_num_experts_for_profiling(model_cfg: ModelConfig) -> Optional[int]:
    if model_cfg.num_experts and model_cfg.num_experts > 128:
        return 128
    return model_cfg.num_experts


def get_prof_dir(cfg: SearchConfig, re_profile=False) -> str:
    if cfg is None:
        return ""
    prof_dir = "auto_settings_profiling"
    prof_dir += f"_{cfg.tp}tp"
    prof_dir += f"_{cfg.dp}dp"
    prof_dir += f"_{cfg.pp}pp"
    prof_dir += f"_{cfg.cp}cp"
    prof_dir += f"_{cfg.mbs}mbs"
    if cfg.is_moe():
        prof_dir += f"_{cfg.ep}ep"
        prof_dir += f"_{cfg.num_experts}experts"
    if cfg.use_ascend_mc2:
        prof_dir += f"_mc2"
    prof_dir += f"_{cfg.seq_length}seq"
    if re_profile:
        prof_dir += f"_re_profile"
    return prof_dir


def get_black_prof_file(config: SearchConfig, re_profile=False) -> str:
    prof_dir = get_prof_dir(config)
    work_dir = get_system_config().work_dir
    node_rank = get_system_config().node_rank
    file_name = f"PP{config.pp}_TP{config.tp}_DP{config.dp}_CP{config.cp}_UP{config.ulysses_size}_MBS{config.mbs}_VP{config.vpp}_EP{config.ep}_node{node_rank}_MODULE.json"
    return os.path.join(work_dir, prof_dir, file_name)


def get_module_info(file_path, key, sub_key=None):
    try:
        with open(file_path, 'r') as file:
            content = json.loads(file.read())
            if sub_key is None:
                return content[key]
            else:
                return content[key][sub_key]
    except FileNotFoundError:
        return float('inf')
    except KeyError:
        return float('inf')
    
    
def check_path_is_link(path: str):
    if os.path.islink(os.path.normpath(path)):
        raise ValueError("The path should not be a symbolic link file. "
                         f"Please check the input path:{path}.")
        
        
def check_path_length_lt(path: str, max_path_length=4096):
    path_length = path.__len__()
    if path_length > max_path_length:
        raise ValueError(f"The length of path should not be greater than {max_path_length}, but got {path_length}. "
                         f"Please check the input path within the valid length range:{path[:max_path_length]}.")
        
        
def standardize_path(
    path: str,
    max_path_length=4096,
    check_link=True,
    check_read=True,
    check_write=True
):
    """
    check path
    param: path
    return: data real path after check
    """
    if path:
        path = os.path.realpath(path)
    else:
        return None

    if os.path.exists(path):
        if check_read and not os.access(path, os.R_OK):
            raise RuntimeError(f"File {path} not readable")

        if check_write and not os.access(path, os.W_OK):
            raise RuntimeError(f"File {path} not writable")
    else:
        print(f"Path: {path} not exists")

    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)

    pattern = r'(\.|/|_|-|\s|[~0-9a-zA-Z]|[\u4e00-\u9fa5])+'
    if not re.fullmatch(pattern, path):
        raise RuntimeError(f"Invalid input path: {path}")

    return path
