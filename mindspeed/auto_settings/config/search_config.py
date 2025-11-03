from enum import IntEnum
from typing import Optional
from dataclasses import dataclass

from mindspeed.auto_settings.config.model_config import ModelConfig


class ExecutorFlag(IntEnum):
    PARSE_ARGS = 1
    PARSE_MODEL = 2
    PROFILE = 3
    PROFILE_BLACK = 4

    def __reduce_ex__(self, protocol):
        # Preventing PKL deserialization failure
        return ExecutorFlag, (self.value,)


@dataclass
class SearchConfig(ModelConfig):
    memory: Optional[float] = None
    performance: Optional[float] = None
    profile_type: ExecutorFlag = ExecutorFlag.PROFILE
    prof_file = None
    auto_settings_ranks = None

    def __str__(self) -> str:
        rt = list()
        rt.append(f"{'Data Parallel Size':<30}{self.data_parallel_size}")
        rt.append(f"{'Tensor Parallel Size':<30}{self.tensor_model_parallel_size}")
        rt.append(f"{'Pipeline Parallel Size':<30}{self.pipeline_model_parallel_size}")
        rt.append(f"{'Virtual Pipeline Size':<30}{self.vpp}")
        rt.append(f"{'Context Parallel Size':<30}{self.context_parallel_size}")
        rt.append(f"{'Expert Parallel Size':<30}{self.expert_model_parallel_size}")
        rt.append(f"{'ZeRO1':<30}{self.use_distributed_optimizer}")
        rt.append(f"{'MC2':<30}{self.use_ascend_mc2}")
        rt.append(f"{'Token Rearrange':<30}{self.enable_token_rearrange_opt}")
        rt.append(f"{'Micro Batch Size':<30}{self.micro_batch_size}")
        rt.append(f"{'Recompute layer':<30}{self.recompute_num_layers}")
        return "\n".join(rt)

    def copy_from_config(self, cfg: ModelConfig):
        for k, v in vars(cfg).items():
            setattr(self, k, v)

    def prepare_for_profiling(self) -> None:
        from mindspeed.auto_settings.config.system_config import get_system_config
        system_config = get_system_config()
        self.micro_batch_size = 1
        self.world_size = system_config.world_size
        self.num_layers_per_virtual_pipeline_stage = None
        self.recompute_granularity = "full"
        self.recompute_method = "block"
        self.recompute_num_layers = self.num_layers // self.pp
        self.use_distributed_optimizer = True
        if self.is_moe():
            self.enable_token_rearrange_opt = True

        self.normalize()
        self.global_batch_size = self.dp * self.pp * self.mbs

        self.train_iters = 10
        self.profile = True
        self.profile_step_start = 8
        self.profile_step_end = 9
        self.profile_level = "level1"
        self.profile_with_cpu = True
        self.profile_with_stack = False
        self.profile_with_memory = True
        self.profile_record_shapes = True

    def prepare_for_profiling_black(self) -> None:
        system_config = get_system_config()
        self.world_size = system_config.world_size
        self.num_layers = self.pp
        self.normalize()
        self.global_batch_size = self.pp * self.dp * self.mbs
        self.auto_settings_ranks = '0'
        self.sequence_parallel = True
        self.train_iters = 10
        self.use_ascend_mc2 = False
        self.noop_layers = None
        self.lr_warmup_iters = None
        self.num_layers_per_virtual_pipeline_stage = None
        self.expert_model_parallel_size = None
        self.use_distributed_optimizer = True
        self.profile = True
        self.profile_step_start = 2
        self.profile_step_end = 3
        self.profile_level = "level1"
        # self.profile_ranks
        self.profile_with_cpu = True
        self.profile_with_stack = True
        self.profile_with_memory = True
        self.profile_record_shapes = True

    def normalize(self) -> None:
        self.data_parallel_size = self.world_size // (self.tp * self.cp * self.pp)

    def crop(self):
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = 1
        return self
