# Copyright 2026 Huawei Technologies Co., Ltd.
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

import dataclasses
from typing import Any, Callable, TypedDict

import torch

from megatron.core.transformer import TransformerConfig
from mindspeed.te.pytorch.fp8.constants import Format, FP8Format, TensorKey
from mindspeed.te.pytorch.fp8.reuse import reuse_or_quantize
from mindspeed.te.pytorch.module_typing import FP8RecipeScaling


class Recipe:

    def __init__(self, key, recipe_config: FP8RecipeScaling, shape):
        self.key = key
        self.config: FP8RecipeScaling = recipe_config
        self.shape = shape
        self.fp8_format: FP8Format = getattr(self.config.fp8_format.value, self.key).value

    def __getattr__(self, item):
        if hasattr(self.__dict__, str(item)):
            return self.__dict__[item]
        return getattr(self.config, str(item))

    @property
    def fp8_format_dtype(self) -> torch.dtype:
        return self.fp8_format.dtype

    @property
    def quant_dtype(self) -> torch.dtype:
        return self.fp8_format.quant_type

    def run_quantizer(
        self,
        tensor: torch.Tensor,
        tensor_key: TensorKey,
        quantizer: Callable[..., Any],
        *,
        allow_reuse: bool = True,
        op_name: str | None = None,
        reuse_identity: Any = None,
        **kwargs: Any,
    ) -> Any:
        return reuse_or_quantize(
            tensor,
            tensor_key,
            quantizer,
            allow_reuse=allow_reuse,
            op_name=op_name,
            reuse_identity=reuse_identity,
            **kwargs,
        )

    def quantization(self, tensor: torch.Tensor, key: TensorKey, colwise: bool, rowwise: bool):
        pass

    def dequantization(self, tensor):
        # Dequantization is implemented inside the kernel path.
        pass


@dataclasses.dataclass
class RecipeScaling:
    recipe = Recipe
    fp8_format: Format
    config: TransformerConfig = None
    fp8_comm: bool = False


class BlockDim(TypedDict):
    row_block_size: int
    col_block_size: int
