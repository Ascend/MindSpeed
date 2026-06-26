# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
from mindspeed.core.optimizer.utils import _distributed_group_rank


def load_parameter_state(self, filename: str, *, update_legacy_format=False):
    """Load distributed optimizer parameter state through CPU for cross-device resume."""
    if self.is_stub_optimizer:
        return
    state_dict = None
    if _distributed_group_rank(self.data_parallel_group) == 0:
        state_dict = torch.load(filename, map_location='cpu')

    self.load_parameter_state_from_dp_zero(state_dict, update_legacy_format=update_legacy_format)
