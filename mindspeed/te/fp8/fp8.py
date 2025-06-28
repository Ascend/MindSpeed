# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Optional
from contextlib import contextmanager
import torch

from mindspeed.te.fp8.state_manager import FP8GlobalStateManager
from mindspeed.te.fp8.recipes.recipe import Recipe


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    fp8_config=None,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[torch.distributed.ProcessGroup] = None,
    _graph: bool = False,
):
    fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
    FP8GlobalStateManager.fp8_autocast_enter(
        enabled=enabled,
        fp8_config=fp8_config,
        calibrating=calibrating,
        fp8_group=fp8_group,
        _graph=_graph,
    )
    try:
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
        FP8GlobalStateManager.fp8_autocast_exit(enabled, _graph=_graph)


