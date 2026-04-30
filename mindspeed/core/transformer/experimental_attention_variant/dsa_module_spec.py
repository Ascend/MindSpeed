# Copyright (c) 2025; Huawei Technologies Co., Ltd.  All rights reserved.

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec


def get_experimental_attention_variant_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Replacement for Megatron's get_experimental_attention_variant_module_spec.

    Adds DSA support while preserving gated_delta_net handling.
    """
    if backend is None:
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            _get_backend_spec_provider,
        )
        backend = _get_backend_spec_provider(config=config)

    if config.experimental_attention_variant == "gated_delta_net":
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_gated_delta_net_module_spec,
        )
        return get_gated_delta_net_module_spec(config=config, backend=backend)
    elif config.experimental_attention_variant == "dsa":
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_dsa_module_spec_for_backend,
        )
        attention = get_dsa_module_spec_for_backend(config=config, backend=backend)
        if "fuse_input_layernorm" not in attention.metainfo:
            attention.metainfo["fuse_input_layernorm"] = False
        return attention
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )
