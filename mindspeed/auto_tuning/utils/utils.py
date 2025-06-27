# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional

from mindspeed.auto_tuning.mindspeed_adaptor.mindspeed_settings import MindSpeedSettings as Settings
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig


def get_tp_for_profiling() -> int:
    tp = Settings().profiling_world_size // 4
    return min(tp, 4)


def get_seq_length_for_profiling(model_cfg: ModelConfig) -> int:
    if not Settings.DISABLE_CP:
        return model_cfg.seq_length
    return min(model_cfg.seq_length, 32 * 1024)


def get_num_experts_for_profiling(model_cfg: ModelConfig) -> Optional[int]:
    if model_cfg.num_experts and model_cfg.num_experts > 128:
        return 128
    return model_cfg.num_experts


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


class NumberConstant:
    """
    Constant for number
    """
    CONVERSION_TIME = 1000.0
    FW_NORM_OP_NUM_DISABLE_PP = 3
    BW_NORM_OP_NUM_DISABLE_PP = 3
    FW_NORM_OP_NUM_ENABLE_PP_LAST_STAGE = 3
    FW_NORM_OP_NUM_ENABLE_PP_OTHER_STAGE = 2

    @property
    def conversion_time(self: any) -> float:
        """
        time conversion us to ms
        :return: time conversion
        """
        return self.CONVERSION_TIME
