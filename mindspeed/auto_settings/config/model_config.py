import math
from typing import Optional
from dataclasses import dataclass

from mindspeed.auto_settings.utils.dtype import DTYPE


@dataclass
class ModelConfig:
    # All params default to None so that errors will be raised
    # once calculations are involved with unresolved params.

    # Model configs
    hidden_size: int = None  # type: ignore
    ffn_hidden_size: int = None
    num_query_groups: int = None
    num_attention_heads: int = None
    swiglu: bool = None
    moe_router_topk: int = None
    num_layers: int = None  # type: ignore
    fp16: bool = None  # type: ignore
    bf16: bool = None  # type: ignore
    n_shared_experts: Optional[int] = None
    num_experts: Optional[int] = None
    seq_length: int = None  # type: ignore
    vocab_size: int = None  # type: ignore
    make_vocab_size_divisible_by: int = 1  # type: ignore
    global_batch_size: int = None  # type: ignore
    micro_batch_size: int = None  # type: ignore
    lr_warmup_iters: bool = None

    # Parallel configs
    world_size: int = None  # type: ignore
    tensor_model_parallel_size: int = None  # type: ignore
    context_parallel_size: int = None  # type: ignore
    pipeline_model_parallel_size: int = None  # type: ignore
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    data_parallel_size: int = None  # type: ignore
    expert_model_parallel_size: int = None  # type: ignore
    sequence_parallel: bool = True

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
    reuse_fp32_param: bool = None
    recompute_activation_function: bool = None
    swap_attention: bool = None
    noop_layers: bool = None


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
    virtual_pipeline_model_parallel_size: int = None
    ring_attention_size: int = None
    ulysses_size: int = None

    def post_init(self):
        if self.num_layers_per_virtual_pipeline_stage:
            self.virtual_pipeline_model_parallel_size = self.num_layers // (
                    self.pp * self.num_layers_per_virtual_pipeline_stage
            )
        if self.virtual_pipeline_model_parallel_size:
            self.num_layers_per_virtual_pipeline_stage = self.num_layers // (
                    self.pp * self.virtual_pipeline_model_parallel_size
            )

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
        return self.virtual_pipeline_model_parallel_size

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
    def padded_vocab_size(self):
        if self.vocab_size:
            division = self.make_vocab_size_divisible_by * self.tensor_model_parallel_size
            padded_vocab_size = int(math.ceil(self.vocab_size / division) * division)
            return padded_vocab_size
        return None

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


_MODEL_CONFIG: ModelConfig = None


def update_model_config(config):
    global _MODEL_CONFIG
    if config is None or _MODEL_CONFIG is None:
        return
    model_config = _MODEL_CONFIG
    for k in vars(config).keys():
        setattr(model_config, k, getattr(config, k, None))
    _MODEL_CONFIG = model_config


def set_model_config(args):
    global _MODEL_CONFIG
    if _MODEL_CONFIG is not None:
        raise AssertionError('MODEL_CONFIG has been initialized')
    model_config = ModelConfig()
    for k in vars(model_config).keys():
        setattr(model_config, k, getattr(args, k, None))
    model_config.post_init()
    _MODEL_CONFIG = model_config


def get_model_config():
    global _MODEL_CONFIG
    if _MODEL_CONFIG is None:
        raise AssertionError('MODEL_CONFIG is not initialized')
    return _MODEL_CONFIG
