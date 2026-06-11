# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""
TTP Replica Chained Optimizer - dynamically injects ChainedOptimizer methods during dump.

Uses __class__ switching to temporarily transform Megatron's native ChainedOptimizer
into TTPReplicaChainedOptimizer during dump, allowing dump args to propagate to sub-optimizers.

No __init__ needed since the optimizer is already initialized at dump time.
"""

import logging

import torch
import torch.distributed as dist

from megatron.core.optimizer.optimizer import ChainedOptimizer

logger = logging.getLogger(__name__)


class TTPReplicaChainedOptimizer(ChainedOptimizer):
    """
    TTP replica chained optimizer, inherits ChainedOptimizer.

    Key modifications:
    1. set_dump_args propagates dump args to the specified sub-optimizer.
    2. save_parameter_state uses the replica group in dump mode.
    3. In dump mode, sharded_state_dict delegates to parent.
    4. Injected via __class__ switching, no re-init needed.
    """

    def set_dump_args(self, optim_idx: int, rank: int, step: int, rank_list: list) -> None:
        """
        Set dump args for a specific sub-optimizer.

        Args:
            optim_idx: Sub-optimizer index.
            rank: Save rank.
            step: Save step count.
            rank_list: Ranks participating in the save.
        """
        if optim_idx >= self.optim_nums:
            raise RuntimeError(f"optim index {optim_idx} out of range [0, {self.optim_nums})")
        self.chained_optimizers[optim_idx].set_dump_args(rank, step, rank_list)

    def need_write_file(self) -> bool:
        """Check if any sub-optimizer needs to write a file"""
        need_write = False
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'need_write_file'):
                need_write |= optimizer.need_write_file()
        return need_write

    def save_parameter_state(self, filename: str) -> None:
        """
        Save parameter state.

        Uses replica group logic in dump mode, original logic in normal mode.
        """
        if len(self.chained_optimizers) == 1:
            optimizer = self.chained_optimizers[0]
            if hasattr(optimizer, 'save_parameter_state'):
                optimizer.save_parameter_state(filename)
            return

        save_states = False
        states = []
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'save_parameter_state_impl'):
                state_dict = optimizer.save_parameter_state_impl()
            elif hasattr(optimizer, 'get_parameter_state_dp_zero'):
                state_dict = optimizer.get_parameter_state_dp_zero()
            else:
                states.append(None)
                continue

            if hasattr(optimizer, 'error_dump') and optimizer.error_dump:
                save_rank = optimizer.save_args.get('rank')
                if dist.get_rank() == save_rank and state_dict is not None:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)
            elif hasattr(optimizer, 'ori_dp_group') and optimizer.ori_dp_group is not None:
                if dist.get_rank(optimizer.ori_dp_group) == 0:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)
            else:
                states.append(state_dict)

        if save_states:
            torch.save(states, filename)

    # sharded_state_dict inherited from parent — no override needed

    def fp32_tensor_to_fp16_tensor(self):
        """Convert fp32 params back to fp16 for all sub-optimizers"""
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'fp32_tensor_to_fp16_tensor'):
                optimizer.fp32_tensor_to_fp16_tensor()
