# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from mindspeed.lite.distributed.fully_shard_parallel.fully_shard_parallel import \
    fully_shard_parallel_modules
from mindspeed.lite.distributed.parallel_state import init_parallel_state
from mindspeed.lite.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from mindspeed.lite.memory.recompute.recompute import recompute_modules
from mindspeed.lite.mindspeed_lite_config import MindSpeedLiteConfig
from mindspeed.lite.distributed.expert_parallel.expert_parallel import expert_parallelize_modules


class MindSpeedLite(torch.nn.Module):
    def __init__(self, config: MindSpeedLiteConfig, model: torch.nn.Module):
        super(MindSpeedLite, self).__init__()
        self.config = config
        self.model = model

        self.parallel_state = init_parallel_state(self.config)
        self.apply_tp_modules()
        self.apply_ep_modules()
        self.apply_recompute_modules()
        self.apply_fsdp_modules()

    def apply_fsdp_modules(self):
        if self.config.fully_shard_parallel_size == 1:
            return
        self.model = fully_shard_parallel_modules(self.model, self.parallel_state.get_fsdp_device_mesh(), self.config.fsdp_plan)

    def apply_tp_modules(self):
        if self.config.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(self.model, self.parallel_state.get_tp_device_mesh(), self.config.tp_plan)

    def apply_ep_modules(self):
        if self.config.expert_parallel_size == 1:
            return
        self.model = expert_parallelize_modules(self.model, self.parallel_state.get_ep_group(), self.config.ep_plan)

    def apply_recompute_modules(self):
        if not self.config.recompute:
            return
        self.model = recompute_modules(self.model, self.config.recompute_plan)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
