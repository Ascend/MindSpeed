from dataclasses import dataclass
from typing import Type, Tuple, Optional

from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling
from mindspeed.te.pytorch.fp8.state_manager import FP8GlobalStateManager


@dataclass
class FP8Config:
    default: Tuple[Type[Recipe], RecipeScaling]
    inputs: Optional[Tuple[Type[Recipe], RecipeScaling]] = None
    weight: Optional[Tuple[Type[Recipe], RecipeScaling]] = None
    grads: Optional[Tuple[Type[Recipe], RecipeScaling]] = None


class FP8Metadata:
    def __init__(self, keys):
        for key in keys:
            setattr(self, key, None)
        self.fp8_recipe_tmp = None

    @property
    def fp8_recipe(self):
        if FP8GlobalStateManager.FP8_RECIPE is not None:
            self.fp8_recipe_tmp = FP8GlobalStateManager.get_fp8_recipe()
        return self.fp8_recipe_tmp

    @property
    def fp8_enable(self):
        return FP8GlobalStateManager.FP8_ENABLED

    @property
    def fusion_matmul(self):
        return FP8GlobalStateManager.FUSION_MATMUL

    @staticmethod
    def create_recipe(key, config: RecipeScaling, tensor_shape):
        return config.recipe(key, config, tensor_shape)

    @staticmethod
    def is_fp8_enable():
        return FP8GlobalStateManager.is_fp8_enabled()

    def init_recipes_if_necessarily(self, key, tensor_shape=None):
        if getattr(self, key) is not None:
            return
        recipe = self.create_recipe(key, self.fp8_recipe, tensor_shape)
        setattr(self, key, recipe)

    def pre_communication(self, key, tensor):
        self.init_recipes_if_necessarily(key, tensor.shape)
        recipe = getattr(self, key)
        return tensor if not recipe.fp8_comm else recipe.pre_communication(tensor)

    def pre_compute(self, key, tensor):
        self.init_recipes_if_necessarily(key, tensor.shape)
        recipe = getattr(self, key)
        return tensor if recipe.fp8_comm else recipe.pre_compute(tensor, key)
