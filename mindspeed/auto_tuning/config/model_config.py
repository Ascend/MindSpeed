# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional
from dataclasses import dataclass

from mindspeed.auto_tuning.utils.dtype import DTYPE


@dataclass
class ModelConfig:
    # All params default to None so that errors will be raised
    # once calculations are involved with unresolved params.

    # Model configs
    hidden_size: int = None  # type: ignore
    num_layers: int = None  # type: ignore
    fp16: bool = None  # type: ignore
    bf16: bool = None  # type: ignore
    n_shared_experts: Optional[int] = None
    num_experts: Optional[int] = None
    seq_length: int = None  # type: ignore
    vocab_size: int = None  # type: ignore
    make_vocab_size_divisible_by: int = None  # type: ignore
    global_batch_size: int = None  # type: ignore
    micro_batch_size: int = None  # type: ignore

    # Parallel configs
    world_size: int = None  # type: ignore
    tensor_model_parallel_size: int = None  # type: ignore
    context_parallel_size: int = None  # type: ignore
    pipeline_model_parallel_size: int = None  # type: ignore
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    data_parallel_size: int = None  # type: ignore
    expert_model_parallel_size: int = None  # type: ignore

    # Feature configs
    untie_embeddings_and_output_weights: bool = None  # type: ignore
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    use_distributed_optimizer: bool = None  # type: ignore
    use_ascend_mc2: bool = None  # type: ignore
    moe_grouped_gemm: bool = None  # type: ignore
    moe_tp_extend_ep: bool = None  # type: ignore
    moe_token_dispatcher_type: str = None  # type: ignore
    enable_token_rearrange_opt: bool = None  # type: ignore
    jit_compile: bool = None  # type: ignore

    # Train & Profile configs
    train_iters: int = None  # type: ignore
    profile: bool = None  # type: ignore
    profile_step_start: int = None  # type: ignore
    profile_step_end: int = None  # type: ignore
    profile_level: str = None  # type: ignore
    profile_with_cpu: bool = None  # type: ignore
    profile_with_stack: bool = None  # type: ignore
    profile_with_memory: bool = None  # type: ignore
    profile_record_shapes: bool = None  # type: ignore

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
    def ep(self) -> int:
        return self.expert_model_parallel_size

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
