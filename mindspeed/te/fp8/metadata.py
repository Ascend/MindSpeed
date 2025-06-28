from dataclasses import dataclass
from typing import Type, Tuple

from mindspeed.te.fp8 import set_amax
from mindspeed.te.fp8.recipes.block_scaling_recipe import BlockScalingRecipe
from mindspeed.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed.te.fp8.state_manager import FP8GlobalStateManager


@dataclass
class FP8Config:
    default: Tuple[Type[Recipe], RecipeConfig] = None
    inputs: Tuple[Type[Recipe], RecipeConfig] = None
    weight: Tuple[Type[Recipe], RecipeConfig] = None
    grads: Tuple[Type[Recipe], RecipeConfig] = None


class FP8Metadata:
    def __init__(self, keys):
        for key in keys:
            setattr(self, key, None)
        self.fp8_config_tmp = None

    @property
    def fp8_config(self):
        if FP8GlobalStateManager.FP8_CONFIG is not None:
            self.fp8_config_tmp = FP8GlobalStateManager.get_fp8_config()
        return self.fp8_config_tmp

    @property
    def fp8_enable(self):
        return FP8GlobalStateManager.FP8_ENABLED

    @property
    def fusion_matmul(self):
        return FP8GlobalStateManager.FUSION_MATMUL

    @staticmethod
    def create_recipe(key, config: Tuple[Type[Recipe], RecipeConfig], tensor_shape):
        recipe, recipe_config = config
        return recipe(key, recipe_config, tensor_shape)

    @staticmethod
    def is_fp8_enable():
        return FP8GlobalStateManager.is_fp8_enabled()

    def init_recipes_if_necessarily(self, key, tensor_shape=None):
        if getattr(self, key) is None:
            fp8_config = self.get_fp8_config(key)
            recipe = self.create_recipe(key, fp8_config, tensor_shape)
            setattr(self, key, recipe)

    def get_fp8_config(self, key):
        fp8_config = self.fp8_config.default
        if hasattr(self.fp8_config, key) and getattr(self.fp8_config, key) is not None:
            fp8_config = getattr(self.fp8_config, key)
        return fp8_config

    def pre_communication(self, key, tensor):
        if not self.get_fp8_config(key)[1].fp8_comm:
            return tensor
        self.init_recipes_if_necessarily(key, tensor.shape)
        recipe = getattr(self, key)
        return tensor if not recipe.fp8_comm else recipe.pre_communication(tensor)

    def pre_compute(self, key, tensor):
        if self.get_fp8_config(key)[1].fp8_comm:
            return tensor
        self.init_recipes_if_necessarily(key, tensor.shape)
        recipe = getattr(self, key)

        return tensor if recipe.fp8_comm else recipe.pre_compute(tensor)

