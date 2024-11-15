from typing import List, Tuple
from logging import Logger

from mindspeed.auto_tuning.utils.logger import get_logger
from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.config.search_config import SearchConfig
from mindspeed.auto_tuning.module.memory.static_mem_modeling import StaticMemModeling
from mindspeed.auto_tuning.module.memory.dynamic_mem_modeling import DynamicMemModeling


class MemoryModeling:
    _static_modeling: StaticMemModeling = None  # type: ignore
    _dynamic_modeling: DynamicMemModeling = None  # type: ignore
    _logger: Logger = None  # type: ignore

    def __new__(cls):
        raise NotImplementedError("MemoryModeling is a static class.")

    @classmethod
    def set_model_cfg(cls, model_cfg: ModelConfig) -> None:
        if cls._static_modeling and cls._dynamic_modeling:
            raise ValueError("ModelConfig has yet been set.")
        cls._static_modeling = StaticMemModeling(model_cfg)
        cls._dynamic_modeling = DynamicMemModeling(model_cfg)
        cls._logger = get_logger("memory")

    @classmethod
    def generate_mem_modeling_profiling_list(cls) -> Tuple[List[Tuple[SearchConfig, str]], List[SearchConfig]]:
        return cls._static_modeling.generate_static_mem_profiling_list(), \
            cls._dynamic_modeling.generate_dynamic_mem_profiling_list()

    @classmethod
    def modeling(cls, working_dir: str) -> None:
        cls._static_modeling.model_static_mem(working_dir)
        cls._dynamic_modeling.model_dynamic_mem(working_dir)

    @classmethod
    def estimate(cls, cfg: SearchConfig) -> Tuple[float, float]:
        cls._logger.debug("==========Memory Estimate Summary==========")
        static_mem = cls._static_modeling.cal_static_mem(cfg)
        dynamic_mem, optimizer_peak = \
            cls._dynamic_modeling.cal_dynamic_mem(cfg)
        peak_stage_mem = float(0)
        for stage_id in range(cfg.pp):
            stage_mem = static_mem[stage_id] + dynamic_mem[stage_id]
            peak_stage_mem = max(peak_stage_mem, stage_mem)
            cls._logger.debug(f"== stage_id: {stage_id} ==\n"
                              f"static memory: {static_mem[stage_id]} MB\n"
                              f"dynamic peak memory: {dynamic_mem[stage_id]} MB\n"
                              f"peak memory: {stage_mem} MB")
        optimizer_peak = max([m + optimizer_peak for m in static_mem])
        cls._logger.debug(f"optimizer peak memory: {optimizer_peak} MB")
        cls._logger.debug("==========Memory Estimate Summary End==========")

        return max(peak_stage_mem, optimizer_peak), optimizer_peak
