# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
from mindspeed.core.optimizer.utils import _distributed_group_rank


def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
    """Load chained distributed optimizer parameter states through CPU."""
    if len(self.chained_optimizers) == 1:
        self.chained_optimizers[0].load_parameter_state(filename, update_legacy_format=update_legacy_format)
        return

    states = None
    for idx, optimizer in enumerate(self.chained_optimizers):
        if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
            continue

        if _distributed_group_rank(optimizer.data_parallel_group) == 0 and states is None:
            states = torch.load(filename, map_location='cpu')

        state_dict = states[idx] if states else None
        optimizer.load_parameter_state_from_dp_zero(state_dict, update_legacy_format=update_legacy_format)
