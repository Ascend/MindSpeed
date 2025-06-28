from abc import abstractmethod
from dataclasses import dataclass

import torch.distributed

from mindspeed.te.fp8.constants import Format


@dataclass
class RecipeConfig:
    margin: int = 0
    interval: int = 1
    fp8_format: Format = Format.E4M3
    amax_history_len: int = 10
    amax_compute_algo: str = 'max'
    amax_reduce_group: torch.distributed.ProcessGroup = None
    block_dim: tuple = None
    fp8_comm: bool = False


class Recipe:

    def __init__(self, key, recipe_config: RecipeConfig, shape):
        self.key = key
        self.config = recipe_config
        self.scale = None
        self.shape = shape


    def __getattr__(self, item):
        if hasattr(self.__dict__, str(item)):
            return self.__dict__[item]
        return getattr(self.config, str(item))

    def pre_communication(self, tensor):
        tensor = self.quantization(tensor, self.scale)
        return tensor

    def pre_compute(self, tensor):
        tensor = self.quantization(tensor, self.scale)
        return tensor

    @abstractmethod
    def quantization(self, tensor, scale_data):
        ...

    @abstractmethod
    def dequantization(self, tensor):
        ...
