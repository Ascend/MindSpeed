# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.distributed

from mindspeed.te.pytorch.fp8.recipes.delayed_scaling_recipe import DelayedScalingRecipe


class FP8GlobalStateManager:
    FP8_ENABLED = False
    FP8_RECIPE = None
    FP8_CALIBRATION = False
    FP8_DISTRIBUTED_GROUP = None
    IS_FIRST_FP8_MODULE = False
    FP8_GRAPH_CAPTURING = False
    FP8_AUTOCAST_DEPTH = 0
    FUSION_MATMUL = False

    @classmethod
    def fp8_autocast_enter(cls, enabled, fp8_recipe, calibrating, fp8_group, _graph):
        cls.FP8_ENABLED = enabled
        cls.FP8_RECIPE = fp8_recipe
        cls.FP8_CALIBRATION = calibrating
        cls.FP8_DISTRIBUTED_GROUP = fp8_group
        cls.FP8_GRAPH_CAPTURING = _graph

        if cls.FP8_AUTOCAST_DEPTH == 0:
            cls.IS_FIRST_FP8_MODULE = True
        cls.FP8_AUTOCAST_DEPTH += 1

        if enabled and not cls.is_fp8_available():
            raise AssertionError('Device not support FP8.')

    @classmethod
    def fp8_autocast_exit(cls, enabled, _graph):
        cls.FP8_AUTOCAST_DEPTH -= 1
        # Reduce only the non-FP8 weight modules here.
        # FP8 weight modules are reduced at the end of the optimizer
        # step after the weight amax is populated.
        if enabled and cls.FP8_AUTOCAST_DEPTH == 0 and not _graph and torch.is_grad_enabled():
            for recipe in DelayedScalingRecipe.ALL_SCALING:
                recipe.finally_step()

    @classmethod
    def get_fp8_autocast_state(cls):
        """FP8 autocast state getter"""
        return (cls.FP8_ENABLED, cls.FP8_RECIPE, cls.FP8_CALIBRATION, cls.FP8_DISTRIBUTED_GROUP,
                cls.IS_FIRST_FP8_MODULE, cls.FP8_GRAPH_CAPTURING,)

    @classmethod
    def set_fp8_autocast_state(cls, fp8_state):
        """FP8 autocast state setter"""
        (cls.FP8_ENABLED, cls.FP8_RECIPE, cls.FP8_CALIBRATION, cls.FP8_DISTRIBUTED_GROUP, cls.IS_FIRST_FP8_MODULE,
         cls.FP8_GRAPH_CAPTURING,) = fp8_state

    @classmethod
    def is_fp8_available(cls) -> bool:
        return True

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        return cls.FP8_ENABLED

    @classmethod
    def get_fp8_recipe(cls):
        return cls.FP8_RECIPE
