import torch

from megatron.core.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, \
    get_tensor_model_parallel_group
from mindspeed.te.pytorch.fp8.constants import TensorKey
from mindspeed.te.pytorch.fp8.state_manager import FP8GlobalStateManager
from mindspeed.te.pytorch.module_typing import FP8Recipe, FP8RecipeScaling, FP8Tensor


class FP8Metadata:
    def __init__(self, keys=None):
        if keys is None:
            keys = [TensorKey.inputs, TensorKey.weight, TensorKey.grads]
        for key in keys:
            setattr(self, key, None)
        self.fp8_recipe_tmp = None
        self.tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tensor_model_parallel_group()

    @property
    def hcom_name(self):
        """通信域handle名"""
        from mindspeed.te.pytorch.utils import get_hccl_comm_name
        return get_hccl_comm_name(self.tp_group, self.tp_rank)

    @property
    def fp8_recipe(self) -> FP8RecipeScaling:
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
    def create_recipe(key: TensorKey, config: FP8RecipeScaling, tensor_shape) -> FP8Recipe:
        return config.recipe(key, config, tensor_shape)

    @staticmethod
    def is_fp8_enable():
        return FP8GlobalStateManager.is_fp8_enabled()

    def init_recipes_if_necessarily(self, key, tensor_shape=None):
        if getattr(self, key) is not None:
            return
        recipe = self.create_recipe(key, self.fp8_recipe, tensor_shape)
        setattr(self, key, recipe)

    def quantization(self, key: TensorKey, tensor: torch.Tensor, colwise=True, rowwise=True) -> FP8Tensor:
        self.init_recipes_if_necessarily(key, tensor.shape)
        recipe: FP8Recipe = getattr(self, key)
        return recipe.quantization(tensor, key, colwise=colwise, rowwise=rowwise)

    def set_tp_config(self, world_size, tp_rank, tp_group):
        self.tp_world_size = world_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
