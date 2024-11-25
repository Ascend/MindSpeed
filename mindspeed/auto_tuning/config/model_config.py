from typing import List, Optional
from dataclasses import dataclass

from mindspeed.auto_tuning.utils.dtype import DTYPE


@dataclass
class ModelConfig:
    ARGS_PARSE_FILENAME = "auto_tuning_model_args.json"
    # Set all parameter defaults to None, so that errors will occur when calculations are performed with
    # unresolved parameters, reflect issues in time.
    # Parallel configs
    tensor_model_parallel_size: int = None  # type: ignore
    context_parallel_size: int = None  # type: ignore
    pipeline_model_parallel_size: int = None  # type: ignore
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    data_parallel_size: int = None  # type: ignore
    sequence_parallel: bool = None  # type: ignore
    use_distributed_optimizer: bool = None  # type: ignore
    global_batch_size: int = None  # type: ignore
    micro_batch_size: int = None  # type: ignore

    # Model configs
    num_layers: int = None  # type: ignore
    num_attention_heads: int = None  # type: ignore
    hidden_size: int = None  # type: ignore
    ffn_hidden_size: int = None  # type: ignore
    add_bias_linear: bool = None  # type: ignore
    swiglu: bool = None  # type: ignore
    fp16: bool = None  # type: ignore
    bf16: bool = None  # type: ignore
    use_ascend_mc2: bool = None  # type: ignore

    # Data configs
    seq_length: int = None  # type: ignore

    # MoE configs
    num_experts: Optional[int] = None
    moe_router_topk: Optional[int] = None
    moe_train_capacity_factor: Optional[float] = None
    expert_model_parallel_size: Optional[int] = None
    enable_token_rearrange_opt: bool = None  # type: ignore

    # Memory configs
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    use_flash_attn: bool = None  # type: ignore
    adaptive_recompute_device_swap: bool = None  # type: ignore

    # Train configs
    train_iters: int = None  # type: ignore
    profile: bool = None  # type: ignore
    profile_step_start: int = None  # type: ignore
    profile_step_end: int = None  # type: ignore
    profile_ranks: List[int] = None  # type: ignore
    profile_level: str = None  # type: ignore
    profile_with_cpu: bool = None  # type: ignore
    profile_with_stack: bool = None  # type: ignore
    profile_with_memory: bool = None  # type: ignore
    profile_record_shapes: bool = None  # type: ignore

    # World Size
    global_world_size: int = None  # type: ignore

    # JIT
    jit_compile: bool = None  # type: ignore

    # Flags
    disable_cp_flag: bool = False

    def __str__(self) -> str:
        rt = list()
        rt.append(f"{'Data Parallel Size':<30}{str(self.dp):<40}")
        rt.append(f"{'Tensor Parallel Size':<30}{str(self.tp):<40}")
        rt.append(f"{'Pipeline Parallel Size':<30}{str(self.pp):<40}")
        rt.append(f"{'Virtual Pipeline Size':<30}{str(self.vpp):<40}")
        rt.append(f"{'Context Parallel Size':<30}{str(self.cp):<40}")
        rt.append(f"{'Expert Parallel Size':<30}{str(self.ep):<40}")
        rt.append(f"{'ZeRO1':<30}{str(self.zero1):<40}")
        rt.append(f"{'MC2':<30}{str(self.use_ascend_mc2):<40}")
        rt.append(f"{'Token Rearrange':<30}{str(self.enable_token_rearrange_opt):<40}")
        rt.append(f"{'Micro Batch Size':<30}{str(self.mbs):<40}")
        rt.append(f"{'Recompute layer':<30}{str(self.re_layer):<40}")
        return "\n".join(rt)

    @property
    def tp(self) -> int:
        return self.tensor_model_parallel_size

    @property
    def cp(self) -> int:
        return self.context_parallel_size

    @property
    def pp(self) -> int:
        return self.pipeline_model_parallel_size

    @property
    def layers_per_vpp(self) -> Optional[int]:
        return self.num_layers_per_virtual_pipeline_stage

    @property
    def vpp(self) -> Optional[int]:
        if self.num_layers_per_virtual_pipeline_stage:
            return self.num_layers // (self.pp * self.num_layers_per_virtual_pipeline_stage)
        return None

    @property
    def dp(self) -> int:
        return self.data_parallel_size

    @property
    def ep(self) -> Optional[int]:
        return self.expert_model_parallel_size or 1

    @property
    def zero1(self) -> bool:
        return self.use_distributed_optimizer

    @property
    def gbs(self) -> int:
        return self.global_batch_size

    @property
    def mbs(self) -> int:
        return self.micro_batch_size

    @property
    def adaptive_recompute(self) -> bool:
        return self.adaptive_recompute_device_swap

    @property
    def re_layer(self) -> Optional[int]:
        return self.recompute_num_layers

    @property
    def num_micro_batches(self) -> int:
        return self.global_batch_size // self.micro_batch_size

    @property
    def dtype(self) -> DTYPE:
        if self.fp16:
            return DTYPE.fp16
        elif self.bf16:
            return DTYPE.bf16
        return DTYPE.fp32

    def is_full_recompute(self) -> bool:
        return self.recompute_granularity is not None and \
            self.recompute_granularity == "full" and \
            self.recompute_method is not None and \
            self.recompute_method == "block"

    def is_moe(self) -> bool:
        return self.num_experts is not None
