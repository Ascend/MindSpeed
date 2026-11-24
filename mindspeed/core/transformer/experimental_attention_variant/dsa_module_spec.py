# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def get_experimental_attention_variant_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Build the module spec for MindSpeed-supported experimental attention variants."""
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

    if config.experimental_attention_variant == "dsa":
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_dsa_module_spec_for_backend,
        )

        attention = get_dsa_module_spec_for_backend(config=config, backend=backend)
        attention.metainfo.setdefault("fuse_input_layernorm", False)
        return attention

    raise ValueError(f"Invalid experimental attention variant: {config.experimental_attention_variant}")
