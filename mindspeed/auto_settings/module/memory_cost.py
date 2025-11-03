"""
算子预估
"""
from mindspeed.auto_settings.config.model_config import get_model_config
from mindspeed.auto_settings.config.system_config import get_system_config
from mindspeed.auto_settings.module.memory.memory_modeling import MemoryModeling
from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.utils import get_num_warmup_micro_batches


class MemoryCost(object):

    def __init__(self):
        self.logger = get_logger("Memory")
        self.memory_model: MemoryModeling = None

    def train_models(self):
        work_dir = get_system_config().work_dir
        self.memory_model = MemoryModeling(get_model_config())
        self.memory_model.modeling(working_dir=work_dir)

    def get_memory_cost(self, config):
        """
        算子执行耗时
        """
        if config.layers_per_vpp:
            layers_per_vpp = config.layers_per_vpp
        else:
            layers_per_vpp = get_model_config().num_layers // config.pp
        num_layers = get_model_config().num_layers // config.pp
        device_mem_cap = get_system_config().memory_cap
        recompute_mem, peak_stage_mem, optimizer_peak = self.memory_model.estimate(config)
        peak_mem = max(peak_stage_mem, optimizer_peak)

        self.logger.debug(f"before recompute, memory = {peak_stage_mem}")
        memory_per_layer = recompute_mem
        warmup_micro_batchs, total_num_micro_batches = get_num_warmup_micro_batches(config, get_model_config())
        release_mem = 0
        max_release_mem = warmup_micro_batchs * layers_per_vpp * memory_per_layer - memory_per_layer
        oom_cap = device_mem_cap - peak_stage_mem
        if max_release_mem <= oom_cap:
            return {
                "layer_calculate": 0,
                "peak_memory": peak_mem,
                "need_recompute": False,
                "new_memory": max_release_mem + peak_stage_mem,
                "recompute_layer": 0
            }
        if config.layers_per_vpp:
            layer_calculate = (oom_cap // (memory_per_layer * config.pp))
            release_mem += layer_calculate * memory_per_layer * config.pp
            if 0 < layer_calculate < num_layers:
                release_mem -= memory_per_layer
            return {
                "layer_calculate": layer_calculate,
                "peak_memory": peak_mem,
                "need_recompute": True,
                "new_memory": release_mem + peak_stage_mem,
                "recompute_layer": num_layers - layer_calculate
            }

        layer_calculate = (oom_cap // (memory_per_layer * config.pp))
        release_mem += layer_calculate * memory_per_layer * config.pp
        if 0 < layer_calculate < num_layers:
            release_mem -= memory_per_layer
        return {
            "layer_calculate": layer_calculate,
            "peak_memory": peak_mem,
            "need_recompute": True,
            "new_memory": release_mem + peak_stage_mem,
            "recompute_layer": num_layers - layer_calculate
        }
