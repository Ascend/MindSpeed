# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Callable

import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig


def megatron_optimizer_init(
    self,
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig,
    init_state_fn: Callable = lambda x: None,
):
    """Input optimizer is the base optimizer (e.g., Adam)."""
    self.optimizer = optimizer
    assert self.optimizer, 'no optimizer is provided.'
    self.empty_optmizer = False
    if getattr(self.optimizer.param_groups[0]['params'][0], 'fake', False):
        self.empty_optmizer = True
    self.config = config
    self.init_state_fn = init_state_fn