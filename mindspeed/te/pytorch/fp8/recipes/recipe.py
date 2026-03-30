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
        **kwargs: Any,
    ) -> Any:
        return reuse_or_quantize(
            tensor,
            tensor_key,
            quantizer,
            allow_reuse=allow_reuse,
            op_name=op_name,
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
