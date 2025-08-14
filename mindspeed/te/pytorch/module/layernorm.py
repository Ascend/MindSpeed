# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch.nn as nn


class MindSpeedTELayernorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps=1e-5, sequence_parallel=False, zero_centered_gamma=False, **kwargs):
        super(MindSpeedTELayernorm, self).__init__(hidden_size, eps=eps)
        self.sequence_parallel = sequence_parallel
        self.zero_centered_gamma = zero_centered_gamma
        setattr(self.weight, 'sequence_parallel', sequence_parallel)
        setattr(self.bias, 'sequence_parallel', sequence_parallel)
        if self.zero_centered_gamma:
            raise NotImplementedError("Zero-centered gamma is not supported in this dummy implementation.")
