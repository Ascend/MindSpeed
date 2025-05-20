# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from .adaptor import (
    multi_latent_attention_init_impl,
    dot_product_attention_init_wrapper,
)

__all__ = [
    "multi_latent_attention_init_impl",
    "dot_product_attention_init_wrapper",
]