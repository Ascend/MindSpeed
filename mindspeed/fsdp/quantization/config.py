# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch_npu


class ScalingStrategyEnum(Enum):
    """
    Quantization strategy — trainable and QAT (Quantization-Aware Training) will be added later.
    """

    DYNAMIC = "dynamic"
    DELAYED = "delayed"
    TRAINABLE = "trainable"
    QAT_W4A16 = "qat—w4a16"


class ScalingGranularityEnum(Enum):
    """
    Quantization granularity
    """

    PER_TENSOR = "pertensor"
    PER_CHANNEL = "perchannel"
    PER_GROUP = "pergroup"
    PER_BLOCK = "perblock"
    MX = "mx"  # microscaling


@dataclass
class ScalingGranularity:
    """
    Quantization granularity
    """

    stype: ScalingGranularityEnum
    block_size: Optional[List[int]]  # [token_dim_block_size, input_dim_block_size, output_dim_block_size]

    def __post_init__(self):
        if self.stype == ScalingGranularityEnum.MX:
            if self.block_size is None:
                self.block_size = [1, 1, 32]
            elif self.block_size != [1, 1, 32]:
                raise ValueError(
                    f"Invalid block size: {self.block_size} for MX, [1, 1, 32] is the only supported block size for MX")


@dataclass
class QuantRecipe:
    """
    Quantization recipe that defines the quantization strategy,
    granularity, and data types for inputs, weights, and gradients.
    """

    scaling_strategy: ScalingStrategyEnum = field(default=ScalingStrategyEnum.DYNAMIC)
    scaling_granularity: ScalingGranularity = field(default_factory=ScalingGranularity)

    inputs_dtype: Optional[torch.dtype] = None
    weight_dtype: Optional[torch.dtype] = None
    grads_dtype: Optional[torch.dtype] = None

    @classmethod
    def from_recipe_name(cls, recipe_name: str):
        """
        Create a QuantRecipe instance based on the recipe name.

        Args:
            recipe_name (str): Name of the quantization recipe.
        Returns：
        QuantRecipe：The corresponding quantization recipe instance。
        """
        if recipe_name not in _registry_quant_recipes:
            # Automatically parse recipe format：
            # <scaling_strategy>_<scaling_granularity>[-blocksize0-blocksize1-blocksize2]_<inputs_dtype>_<weight_dtype>_<grads_dtype>
            # eg：
            # MXFP8：dynamic_MX-1-1-32_E4M3_E4M3_E4M3
            # DeepSeek FP8: dynamic_blockwise-1-128-128_E4M3_E4M3_E4M3
            # QAT W4A16(MXFP4): qat-w4a16_MX-1-1-32_BF16_E2M1_BF16

            parts = recipe_name.split("_")
            if len(parts) == 5:
                scaling_strategy, scaling_granularity, inputs_dtype, weight_dtype, grads_dtype = parts
                try:
                    scaling_strategy = ScalingStrategyEnum(scaling_strategy)

                    if "-" in scaling_granularity:
                        scaling_granularity, block_size = scaling_granularity.split("-")[0], scaling_granularity.split(
                            "-")[1:]
                        if len(block_size) != 3:
                            raise ValueError(f"Invalid block size: {block_size}")

                        block_size = [int(b) for b in block_size]
                    else:
                        block_size = None

                    scaling_granularity = ScalingGranularity(ScalingGranularityEnum(scaling_granularity), block_size)

                    inputs_dtype = _dtype_mapping[inputs_dtype]
                    weight_dtype = _dtype_mapping[weight_dtype]
                    grads_dtype = _dtype_mapping[grads_dtype]
                    return cls(
                        scaling_strategy=scaling_strategy,
                        scaling_granularity=scaling_granularity,
                        inputs_dtype=inputs_dtype,
                        weight_dtype=weight_dtype,
                        grads_dtype=grads_dtype,
                    )

                except ValueError:
                    pass
            raise ValueError(f"Unknown recipe name: {recipe_name}")

        return _registry_quant_recipes[recipe_name]

    def get_key_dtype(self, key: str):
        if key == "inputs":
            return self.inputs_dtype
        elif key == "weight":
            return self.weight_dtype
        elif key == "grads":
            return self.grads_dtype
        else:
            raise ValueError(f"Unknown key: {key}")


_registry_quant_recipes: Dict[str, QuantRecipe] = dict()
_dtype_mapping = {
    "E1M2": torch_npu.float4_e1m2fn_x2,
    "E2M1": torch_npu.float4_e2m1fn_x2,
    "E4M3": torch.float8_e4m3fn,
    "E5M2": torch.float8_e5m2,
    "HiF8": torch_npu.hifloat8,
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
}


def recipe_register(name=None):
    """
    Register a quantization recipe into the `recipes` dictionary.
    Args:
        name (str, optional): The name of the recipe.
        If not provided, the name of the decorated function will be used.
    """

    def decorator(obj):
        key = name if name is not None else obj.__name__
        _registry_quant_recipes[key] = obj
        return obj

    # Handle the case of @register without arguments (callable directly)
    if callable(name):
        obj = name
        name = None
        return decorator(obj)
    # Handle case of @register('name') or @register(name='name') with arguments
    else:
        return decorator


@recipe_register("mxfp8")
def register_mxfp8_recipe():
    """Register a MXFP8 recipe to the recipes dictionary."""
    return QuantRecipe(
        scaling_strategy=ScalingStrategyEnum.DYNAMIC,
        scaling_granularity=ScalingGranularity(ScalingGranularityEnum.MX, [1, 1, 32]),
        inputs_dtype=_dtype_mapping["E4M3"],
        weight_dtype=_dtype_mapping["E4M3"],
        grads_dtype=_dtype_mapping["E4M3"],
    )


@recipe_register("delayed_hif8_pertensor")
def register_delayed_hif8_per_tensor_recipe():
    """Register a per-tensor FP8 delayed recipe to the recipes dictionary."""
    return QuantRecipe(
        scaling_strategy=ScalingStrategyEnum.DELAYED,
        scaling_granularity=ScalingGranularity(ScalingGranularityEnum.PER_TENSOR),
        inputs_dtype=_dtype_mapping["HiF8"],
        weight_dtype=_dtype_mapping["HiF8"],
        grads_dtype=_dtype_mapping["HiF8"],
    )
