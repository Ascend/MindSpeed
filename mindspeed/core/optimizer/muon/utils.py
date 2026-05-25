# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch


def get_pg_size(group: Optional[torch.distributed.ProcessGroup]) -> int:
    """Return process group size, treating None as a single-rank group."""
    if group is None:
        return 1
    return torch.distributed.get_world_size(group=group)


def get_pg_rank(group: Optional[torch.distributed.ProcessGroup]) -> int:
    """Return process group rank, treating None as a single-rank group."""
    if group is None:
        return 0
    return torch.distributed.get_rank(group=group)


class LegacyProcessGroupCollection:
    """Process-group collection with the field names used by Megatron dev."""

    def __init__(self) -> None:
        from megatron.core import mpu

        self.mp = mpu.get_model_parallel_group()
        self.tp = mpu.get_tensor_model_parallel_group()
        try:
            self.dp_cp = mpu.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)
        except TypeError:
            self.dp_cp = mpu.get_data_parallel_group()

        if hasattr(mpu, "get_expert_tensor_parallel_group"):
            try:
                self.expt_tp = mpu.get_expert_tensor_parallel_group()
            except (AssertionError, RuntimeError):
                self.expt_tp = self.tp
        else:
            self.expt_tp = self.tp

        if hasattr(mpu, "get_expert_data_parallel_group"):
            try:
                self.expt_dp = mpu.get_expert_data_parallel_group()
            except (AssertionError, RuntimeError):
                self.expt_dp = self.dp_cp
        else:
            self.expt_dp = self.dp_cp

        for getter_name in (
            "get_expert_tensor_model_pipeline_parallel_group",
            "get_expert_tensor_and_model_parallel_group",
        ):
            if hasattr(mpu, getter_name):
                try:
                    self.tp_ep_pp = getattr(mpu, getter_name)()
                    break
                except (AssertionError, RuntimeError):
                    continue
        else:
            self.tp_ep_pp = self.mp
