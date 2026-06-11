# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""
TTP FP32 Replica Optimizer - dynamically injects FP32Optimizer methods during dump.

Uses __class__ switching to temporarily transform Megatron's native FP32Optimizer
into TTPFP32ReplicaOptimizer during dump.

No __init__ needed since the optimizer is already initialized at dump time.
"""

import logging

import torch.distributed as dist

from megatron.core.optimizer.optimizer import FP32Optimizer

logger = logging.getLogger(__name__)


class TTPFP32ReplicaOptimizer(FP32Optimizer):
    """
    TTP FP32 replica optimizer, inherits FP32Optimizer.

    Key modifications:
    1. In dump mode, sharded_state_dict returns None (FP32 does not participate in dist ckpt).
    2. Provides set_dump_args / need_write_file interface.
    3. Injected via __class__ switching, no re-init needed.
    """

    def set_dump_args(self, rank: int, step: int, rank_list: list) -> None:
        """Set dump args, switching the optimizer to dump mode"""
        self.save_args['step'] = step
        self.save_args['rank'] = rank
        self.save_args['rank_list'] = rank_list
        self._error_dump = True

    @property
    def error_dump(self):
        return getattr(self, '_error_dump', False)

    @error_dump.setter
    def error_dump(self, value):
        self._error_dump = value

    def need_write_file(self) -> bool:
        """Check if the current rank needs to write a file"""
        cur_rank = dist.get_rank()
        if self.error_dump and self.save_args.get('rank') == cur_rank:
            return True
        if not self.error_dump and self.ori_dp_group is not None:
            return dist.get_rank(self.ori_dp_group) == 0
        return False

    def sharded_state_dict(self, model_sharded_state_dict, is_loading, **kwargs):  # noqa: W0246
        """
        Consistent with MindSpeed TTP design: FP32 optimizer returns None and
        does not participate in distributed checkpoint saving.
        """
        pass

    def save_parameter_state_impl(self):
        """FP32 optimizer has no DistributedOptimizer param state, return None"""
        return None

    def fp32_tensor_to_fp16_tensor(self):
        """No-op"""
        pass
