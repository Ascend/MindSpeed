# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    args: any

    model_config: dict = field(init={})

    vocab_size: int = field(init=0)

    def __post_init__(self):
        self.vocab_size = self.args.padded_vocab_size

        if hasattr(self.args, 'moe_intermediate_size') and self.args.moe_intermediate_size:
            self.args.ffn_hidden_size = self.args.moe_intermediate_size

        if self.args.ffn_hidden_size is None:
            self.args.ffn_hidden_size = 4 * self.args.hidden_size

        if not self.args.group_query_attention:
            self.args.num_query_groups = self.args.num_attention_heads

    def is_moe_model(self):
        return self.args.num_experts is not None


_MODEL_CONFIG: ModelConfig = None


def set_model_config(model_config: ModelConfig):
    global _MODEL_CONFIG
    if _MODEL_CONFIG is not None:
        raise AssertionError('MODEL_CONFIG has been initialized')
    _MODEL_CONFIG = model_config


def get_model_config():
    global _MODEL_CONFIG
    if _MODEL_CONFIG is None:
        raise AssertionError('MODEL_CONFIG is not initialized')
    return _MODEL_CONFIG

