from mindspeed.auto_tuning.module.hardware import Hardware
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig


def get_tp_for_profiling() -> int:
    tp = Hardware().num_devices // 4
    if "910B" in Hardware().device_type:
        tp = min(tp, 8)
    return tp


def get_seq_length_for_profiling(model_cfg: ModelConfig) -> int:
    if model_cfg.disable_cp_flag:
        return model_cfg.seq_length
    return min(model_cfg.seq_length, 32 * 1024)


def get_prof_dir(cfg: SearchConfig, re_profile=False) -> str:
    prof_dir = "auto_tuning_profiling"
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
