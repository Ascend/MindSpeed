from typing import Optional
from dataclasses import dataclass

from mindspeed.auto_tuning.config.model_config import ModelConfig
from mindspeed.auto_tuning.module.hardware import Hardware


@dataclass
class SearchConfig(ModelConfig):
    memory: Optional[float] = None
    performance: Optional[float] = None

    def __str__(self) -> str:
        rt = list()
        if self.performance:
            rt.append(f"{'Performance':<30}{str(self.performance):<40}")
        if self.memory:
            rt.append(f"{'Memory':<30}{str(self.memory):<40}")
        return super().__str__() + "\n" + "\n".join(rt)

    def copy_from_config(self, cfg: ModelConfig) -> None:
        for k, v in vars(cfg).items():
            if k in self.__dict__:
                self.__dict__[k] = v

    def prepare_for_profiling(self) -> None:
        self.use_distributed_optimizer = True
        self.recompute_granularity = "full"
        self.recompute_method = "block"
        self.adaptive_recompute_device_swap = False
        self.global_world_size = Hardware().num_devices
        self.micro_batch_size = 1

        self.normalize()
        self.global_batch_size = self.dp * self.pp * self.mbs

        self.train_iters = 10
        self.profile = True
        self.profile_step_start = 8
        self.profile_step_end = 9
        self.profile_ranks = list(range(Hardware().num_devices))
        self.profile_level = "level1"
        self.profile_with_cpu = True
        self.profile_with_stack = False
        self.profile_with_memory = True
        self.profile_record_shapes = True

    def normalize(self) -> None:
        self.data_parallel_size = self.global_world_size // \
            (self.tp * self.cp * self.pp)

        if self.is_moe():
            self.enable_token_rearrange_opt = True

        if self.adaptive_recompute_device_swap:
            self.recompute_granularity = None
            self.recompute_method = None
            self.recompute_num_layers = None
        elif self.is_full_recompute():
            self.recompute_num_layers = self.num_layers // self.pp
