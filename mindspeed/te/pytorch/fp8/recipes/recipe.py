import dataclasses
from typing import TypedDict

from megatron.core.transformer import TransformerConfig
from mindspeed.te.pytorch.fp8.constants import Format, FP8Format


class Recipe:

    def __init__(self, key, recipe_config: 'RecipeScaling', shape):
        self.key = key
        self.config = recipe_config
        self.shape = shape
        self.fp8_format: FP8Format = getattr(self.config.fp8_format.value, self.key).value

    def __getattr__(self, item):
        if hasattr(self.__dict__, str(item)):
            return self.__dict__[item]
        return getattr(self.config, str(item))

    @property
    def fp8_format_dtype(self):
        return self.fp8_format.dtype

    @property
    def quant_dtype(self):
        return self.fp8_format.quant_type

    def pre_communication(self, tensor, key=None):
        tensor = self.quantization(tensor, key)
        return tensor

    def pre_compute(self, tensor, key=None):
        tensor = self.quantization(tensor, key)
        return tensor

    def quantization(self, tensor, key=None):
        pass

    def dequantization(self, tensor):
        # 算子内实现反量化, 这里不在做实现
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
