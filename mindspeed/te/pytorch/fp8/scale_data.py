import logging
from typing import List

import torch

from mindspeed.te.pytorch.fp8.constants import AMAX_COMPUTE_MAP, FP8Format

logger = logging.getLogger(__name__)


class ScaleData:
    def __init__(self, recipe_config, fp8_format, scale_shape: List[int] = None):
        self.config = recipe_config
        self.ori_dtype = None
        self.scale_shape = scale_shape if scale_shape is not None else [1]
        self.device = 'npu:{}'.format(torch.npu.current_device())

        self.fp8_format: FP8Format = fp8_format
        self.fp8_max = self.fp8_format.max
        self.margin = self.config.config.fp8_margin
        self.scale = torch.ones(self.scale_shape, device=self.device)

        self.amax_history_len = self.config.config.fp8_amax_history_len
        self.amax_history_current_len = 0
        if self.config.config.fp8_amax_compute_algo not in AMAX_COMPUTE_MAP:
            raise AssertionError('Unsupported amax compute algo {}'.format(self.config.config.fp8_amax_compute_algo))
        self.amax_compute = AMAX_COMPUTE_MAP[self.config.config.fp8_amax_compute_algo]
        # 存储结构 -> tensor([amax_len, block])
        self.amax_history = torch.zeros([self.amax_history_len] + self.scale_shape, device=self.device)
        self.amax = torch.zeros(self.scale_shape, device=self.device)
        self.current_interval = 1

    @property
    def quantization_scale(self):
        return self.scale if self.scale.numel() == 1 else self.scale[0][0]

    @property
    def last_history_index(self):
        if self.amax_history_current_len < self.amax_history_len:
            return self.amax_history_current_len - 1
        return -1

    def append_amax(self, amax):
        if self.amax_history_current_len < self.amax_history_len:
            self.amax_history[self.amax_history_current_len, :].copy_(amax, non_blocking=True)
            self.amax_history_current_len += 1
        else:
            self.amax_history = self.amax_history.roll(-1, 1)
            self.amax_history[self.amax_history_len - 1, :].copy_(amax, non_blocking=True)

    def reduce_amax(self, group=None):
        if group is None or torch.distributed.get_world_size(group) <= 1:
            return
        if self.amax_history_current_len < self.amax_history_len:
            amax = self.amax_history[self.amax_history_current_len - 1, :]
        else:
            amax = self.amax_history[self.amax_history_len - 1, :]
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=group)

    def delayed_recipe_update_scale(self):
        self.reduce_amax(self.config.amax_reduce_group)
        self.amax_compute(self.amax, self.amax_history, self.last_history_index)
        # 这里为适配算子对原始公式进行取反
        # 原始公式 (self.fp8_max / self.amax) / (2 ** self.margin)
        self.scale.copy_(((self.amax * (2 ** self.margin)) / self.fp8_max), non_blocking=True)

    def delayed_recipe_update_amax(self, tensor, stream):
        if self.amax_history_current_len == 0:
            # first amax will use current max
            self.current_interval = 1
            amax = torch.amax(torch.abs(tensor))
            self.append_amax(amax)
            self.delayed_recipe_update_scale()
            scale = self.scale.clone()
        else:
            stream.wait_stream(torch.cuda.current_stream())
            scale = self.scale.clone()

            if self.current_interval >= self.config.config.fp8_interval:
                self.current_interval = 1
                with torch.cuda.stream(stream):
                    amax = torch.amax(torch.abs(tensor))
                    self.append_amax(amax)
                    self.delayed_recipe_update_scale()
            else:
                self.current_interval += 1
        return scale
