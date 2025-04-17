import json
import os
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
