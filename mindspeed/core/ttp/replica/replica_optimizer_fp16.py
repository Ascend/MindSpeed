# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""
TTP FP16 Replica Optimizer - dynamically injects Float16OptimizerWithFloat16Params methods during dump.

Uses __class__ switching to temporarily transform Megatron's native Float16OptimizerWithFloat16Params
into TTPFP16ReplicaOptimizer during dump.

No __init__ needed since the optimizer is already initialized at dump time.
"""

import logging

import torch.distributed as dist

from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

logger = logging.getLogger(__name__)


class TTPFP16ReplicaOptimizer(Float16OptimizerWithFloat16Params):
    """
    TTP FP16 replica optimizer, inherits Float16OptimizerWithFloat16Params.

    Key modifications:
    1. In dump mode, sharded_state_dict delegates to parent.
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

    # sharded_state_dict inherited from parent — no override needed

    def save_parameter_state_impl(self):
        """FP16 optimizer has no DistributedOptimizer param state, return None"""
        return None

    def fp32_tensor_to_fp16_tensor(self):
        """No-op; TTP does not require fp32-to-fp16 conversion"""
        pass
