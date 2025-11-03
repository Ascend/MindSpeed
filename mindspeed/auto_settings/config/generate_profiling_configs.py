from typing import List
import json
from pathlib import Path

from mindspeed.auto_settings.config.model_config import ModelConfig
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.config.post_info import PostInfo
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.file_utils import check_file_size
from mindspeed.auto_settings.utils.utils import (
    get_tp_for_profiling, get_seq_length_for_profiling, get_num_experts_for_profiling, get_prof_dir
)

_GEN_CFGS_FILENAME = "profiling_configs.json"
_GEN_CFGS_FILENAME_VL = "profiling_configs_vl.json"

_logger = get_logger("gen_prof_cfgs")


def generate_profiling_configs(setting: PostInfo, model_config) -> List[SearchConfig]:
    profile_cfgs: List[SearchConfig] = list()

    base_tp = get_tp_for_profiling()
    base_seq_len = get_seq_length_for_profiling(model_config)
    base_num_experts = get_num_experts_for_profiling(model_config)

    if setting.dist_train:
        _logger.debug(f"setting.dist_train:")
        with Path(__file__).parent.joinpath(_GEN_CFGS_FILENAME_VL).open(encoding="utf-8") as f:
            check_file_size(f)
            cfgs = json.load(f)
    else:
        _logger.debug(f"setting.is singo model:")
        with Path(__file__).parent.joinpath(_GEN_CFGS_FILENAME).open(encoding="utf-8") as f:
            check_file_size(f)
            cfgs = json.load(f)

    for cfg in cfgs:
        if "skip" in cfg.get("name", ""):
            _logger.debug(f"{cfg} asked to skip.")
            continue

        if setting.DISABLE_CP and cfg.get("cp", 1) > 1:
            _logger.debug(f"Not searching cp, dropped {cfg}.")
            continue

        gen_cfg = SearchConfig()
        gen_cfg.copy_from_config(model_config)

        tp = cfg.get("tp", "default")
        if tp == "default":
            gen_cfg.tensor_model_parallel_size = base_tp
        elif tp.startswith("mul_t_by="):
            gen_cfg.tensor_model_parallel_size = base_tp * int(tp.strip().split("=")[1])
        else:
            raise ValueError(f"Not supporting value on tp field: {tp} of {cfg}.")

        gen_cfg.context_parallel_size = cfg.get("cp", 1)
        gen_cfg.pipeline_model_parallel_size = cfg.get("pp", 1)
        gen_cfg.num_layers = cfg.get("pp", 1)
        gen_cfg.use_ascend_mc2 = cfg.get("mc2", False)

        if "tp" not in model_config.parallel_switch:
            gen_cfg.tensor_model_parallel_size = 1
        
        if "cp" not in model_config.parallel_switch:
            gen_cfg.context_parallel_size = 1

        seq = cfg.get("seq", "default")
        if seq == "default":
            gen_cfg.seq_length = base_seq_len
        elif seq.startswith("slice_seq_by="):
            slice_rate = int(seq.strip().split("=")[1])
            if base_seq_len // slice_rate > 2 * 1024:
                gen_cfg.seq_length = base_seq_len // slice_rate
            else:
                gen_cfg.seq_length = base_seq_len * slice_rate
        else:
            raise ValueError(f"Not supporting value on seq field: {seq} of {cfg}.")

        if model_config.is_moe():
            num_experts = cfg.get("experts", "default")
            if num_experts == "default":
                gen_cfg.num_experts = base_num_experts
            else:
                raise ValueError(f"Not supporting value on experts field: {num_experts} of {cfg}.")
            gen_cfg.expert_model_parallel_size = cfg.get("ep", 1)

        if gen_cfg.seq_length // gen_cfg.cp <= 2 * 1024:
            _logger.debug(f"Seq per cp too small, dropped {cfg}.")
            continue

        gen_cfg.prepare_for_profiling()
        if gen_cfg.ep and gen_cfg.ep > gen_cfg.dp * gen_cfg.cp:
            raise ValueError(f"ep > dp * cp of {cfg}.")

        profile_cfgs.append((gen_cfg, get_prof_dir(cfg=gen_cfg)))

    return profile_cfgs
