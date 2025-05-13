# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch.nn as nn


class MindSpeedTELayernorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, sequence_parallel=False, zero_centered_gamma=False, **kwargs):
        super(MindSpeedTELayernorm, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.sequence_parallel = sequence_parallel
        self.zero_centered_gamma = zero_centered_gamma
        if self.zero_centered_gamma:
            raise NotImplementedError("Zero-centered gamma is not supported in this dummy implementation.")
        if self.sequence_parallel:
            raise NotImplementedError("Sequence parallelism is not supported in this dummy implementation.")

    def forward(self, x):
        return self.layer_norm(x)
