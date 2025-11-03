from typing import List, Tuple
from logging import Logger

from mindspeed.auto_settings.utils.logger import get_logger, change_stream_handler
from mindspeed.auto_settings.config.model_config import ModelConfig, get_model_config
from mindspeed.auto_settings.config.search_config import SearchConfig
from mindspeed.auto_settings.module.memory.static_mem_modeling import StaticMemModeling
from mindspeed.auto_settings.module.memory.dynamic_mem_modeling import DynamicMemModeling


class MemoryModeling:
    def __init__(self, model_cfg: ModelConfig):
        self._static_modeling = StaticMemModeling(model_cfg)
        self._dynamic_modeling = DynamicMemModeling(model_cfg)
        self._logger = get_logger("memory")

    @classmethod
    def set_model_cfg(cls, model_cfg: ModelConfig) -> None:
        model_cfg = get_model_config()
        cls._static_modeling = StaticMemModeling(model_cfg)
        cls._dynamic_modeling = DynamicMemModeling(model_cfg)
        cls._logger = get_logger("memory")

    def generate_mem_modeling_profiling_list(self) -> Tuple[List[Tuple[SearchConfig, str]], List[SearchConfig]]:
        return self._static_modeling.generate_static_mem_profiling_list(), \
            self._dynamic_modeling.generate_dynamic_mem_profiling_list()

    def modeling(self, working_dir: str) -> None:
        self._static_modeling.model_static_mem(working_dir)
        self._dynamic_modeling.model_dynamic_mem(working_dir)

    def estimate(self, cfg: SearchConfig, parallel=False, output=None) -> Tuple[float, float, float]:
        if parallel is True:
            change_stream_handler(self._logger, output)
        self._logger.debug("==========Memory Estimate Summary==========")
        static_mem = self._static_modeling.cal_static_mem(cfg)
        dynamic_mem, recompute_mem, optimizer_peak = \
            self._dynamic_modeling.cal_dynamic_mem(cfg, output)
        peak_stage_mem = float(0)
        for stage_id in range(cfg.pp):
            stage_mem = static_mem[stage_id] + dynamic_mem[stage_id]
            peak_stage_mem = max(peak_stage_mem, stage_mem)
            self._logger.debug(f"== stage_id: {stage_id} ==\n"
                              f"static memory: {static_mem[stage_id]} MB\n"
                              f"dynamic peak memory: {dynamic_mem[stage_id]} MB\n"
                              f"peak memory: {stage_mem} MB")
        optimizer_peak = max([m + optimizer_peak for m in static_mem])
        self._logger.debug(f"optimizer peak memory: {optimizer_peak} MB")
        self._logger.debug("==========Memory Estimate Summary End==========")

        return recompute_mem, peak_stage_mem, optimizer_peak

