from typing import List
from dataclasses import replace

from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.utils.utils import get_tp_for_profiling, get_seq_length_for_profiling


def generate_profiling_configs(model_cfg: ModelConfig) -> List[SearchConfig]:
    profile_cfgs: List[SearchConfig] = list()

    base_cfg = SearchConfig()
    base_cfg.copy_from_config(model_cfg)
    base_cfg.tensor_model_parallel_size = get_tp_for_profiling()
    base_cfg.context_parallel_size = 1
    base_cfg.pipeline_model_parallel_size = 1
    base_cfg.seq_length = get_seq_length_for_profiling(model_cfg)
    if model_cfg.is_moe():
        base_cfg.num_experts = 4
        base_cfg.expert_model_parallel_size = 4
    bi_tp = base_cfg.tp * 2
    if "910B" in Hardware().device_type and base_cfg.tp == 8:
        bi_tp = 4

    if "910_9" in Hardware().device_type and base_cfg.tp == 8:
        bi_tp = 16

    # base config
    # 4dp
    profile_cfgs.append(base_cfg)

    # 4dp mc2
    gen_cfg_mc2 = replace(base_cfg, use_ascend_mc2=True)
    profile_cfgs.append(gen_cfg_mc2)

    # 2dp 2tp
    gen_cfg = replace(base_cfg)
    gen_cfg.tensor_model_parallel_size = bi_tp
    if model_cfg.is_moe():
        gen_cfg.expert_model_parallel_size = 2
    profile_cfgs.append(gen_cfg)

    # 2dp 2tp mc2
    gen_cfg_mc2 = replace(gen_cfg, use_ascend_mc2=True)
    profile_cfgs.append(gen_cfg_mc2)

    # 2dp 2pp
    gen_cfg = replace(base_cfg)
    gen_cfg.pipeline_model_parallel_size = 2
    if model_cfg.is_moe():
        gen_cfg.expert_model_parallel_size = 2
    profile_cfgs.append(gen_cfg)

    # CP config
    if not model_cfg.disable_cp_flag:
        # 4cp
        gen_cfg = replace(base_cfg)
        gen_cfg.context_parallel_size = 4
        if gen_cfg.seq_length // gen_cfg.cp >= 2 * 1024:
            profile_cfgs.append(gen_cfg)

        # 2cp
        gen_cfg = replace(base_cfg)
        gen_cfg.context_parallel_size = 2
        if model_cfg.is_moe():
            gen_cfg.expert_model_parallel_size = 2
        if gen_cfg.seq_length // gen_cfg.cp >= 2 * 1024:
            profile_cfgs.append(gen_cfg)

        # roce cp
        gen_cfg = replace(base_cfg)
        gen_cfg.context_parallel_size = 2
        gen_cfg.tensor_model_parallel_size = bi_tp
        if model_cfg.is_moe():
            gen_cfg.expert_model_parallel_size = 2
        if gen_cfg.seq_length // gen_cfg.cp >= 2 * 1024:
            profile_cfgs.append(gen_cfg)

    # MLP config
    if model_cfg.is_moe():
        gen_cfg = replace(base_cfg)
        gen_cfg.expert_model_parallel_size = 1
        gen_cfg.pipeline_model_parallel_size = 1
        profile_cfgs.append(gen_cfg)

        gen_cfg_pp2 = replace(gen_cfg)
        gen_cfg_pp2.pipeline_model_parallel_size = 2
        profile_cfgs.append(gen_cfg_pp2)

    # half-seq
    gen_cfg = replace(base_cfg)
    if model_cfg.is_moe():
        gen_cfg.expert_model_parallel_size = 1
    gen_cfg.seq_length = base_cfg.seq_length // 2
    if gen_cfg.seq_length < 2 * 1024:
        gen_cfg.seq_length = gen_cfg.seq_length * 4

    for cfg in profile_cfgs:
        cfg.prepare_for_profiling()
        cfg.num_layers = cfg.pp

    return profile_cfgs
