# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import contextmanager
from typing import Optional

import torch

from mindspeed.te.pytorch.fp8.state_manager import FP8GlobalStateManager


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe=None,
    fp8_group: Optional[torch.distributed.ProcessGroup] = None,
    _graph: bool = False,
):
    fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
    FP8GlobalStateManager.fp8_autocast_enter(
        enabled=enabled,
        fp8_recipe=fp8_recipe,
        calibrating=calibrating,
        fp8_group=fp8_group,
        _graph=_graph,
    )
    try:
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
        FP8GlobalStateManager.fp8_autocast_exit(enabled, _graph=_graph)


@contextmanager
def fp8_model_init(
    enabled: bool = True,
    recipe=None,
    preserve_high_precision_init_val: bool = False,
) -> None:
    """
    Context manager for FP8 initialization of parameters.

    Example usage:

    .. code-block:: python

        with fp8_model_init(enabled=True):
            model = transformer_engine.pytorch.Linear(768, 768)

        # Preserving high precision initial value to initialize master weight
        with fp8_model_init(enabled=True, preserve_high_precision_init_val=True):
            model = transformer_engine.pytorch.Linear(768, 768)
        master_weight = model.weight.get_high_precision_init_val()
        model.weight.clear_high_precision_init_val()

    Parameters
    ----------
    enabled: bool, default = `True`
             when enabled, Transformer Engine modules created inside this `fp8_model_init`
             region will hold only FP8 copies of its parameters, as opposed to the default
             behavior where both higher precision and FP8 copies are present. Setting this
             option to `True` may result in lower memory consumption and is especially
             useful for scenarios like:

             * full model training using optimizer with master weights, where the high
               precision copies of weights are already present in the optimizer.
             * inference, where only the FP8 copies of the parameters are used.
             * LoRA-like fine-tuning, where the main parameters of the model do not change.
    recipe: transformer_engine.common.recipe.Recipe, default = `None`
            Recipe used to create the parameters. If left to None, it uses the default FP8 recipe.
    preserve_high_precision_init_val: bool, default = `False`
             when enabled, store the high precision tensor used to initialize FP8 parameters
             in CPU memory, and add two function attributes named `get_high_precision_init_val()`
             and `clear_high_precision_init_val()` to FP8 parameters to get/clear this high
             precision tensor. The purpose is that users can use this high-precision copy
             to initialize master weights, avoiding the loss of precision that can occur when
             using FP8 parameters directly. Note that after the master weights are initialized,
             users should call `clear_high_precision_init_val()` to release this CPU memory.

             This functionality is *EXPERIMENTAL*.
    """
    _fp8_parameters = FP8GlobalStateManager.FP8_PARAMETERS
    _fp8_recipe = FP8GlobalStateManager.FP8_RECIPE
    _high_precision_init_val = FP8GlobalStateManager.HIGH_PRECISION_INIT_VAL
    FP8GlobalStateManager.FP8_PARAMETERS = enabled
    FP8GlobalStateManager.FP8_RECIPE = recipe
    FP8GlobalStateManager.HIGH_PRECISION_INIT_VAL = preserve_high_precision_init_val
    try:
        yield
    finally:
        FP8GlobalStateManager.FP8_PARAMETERS = _fp8_parameters
        FP8GlobalStateManager.FP8_RECIPE = _fp8_recipe
        FP8GlobalStateManager.HIGH_PRECISION_INIT_VAL = _high_precision_init_val
