# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
Sequence 1F1B Partitioning and Queue Management for Sequence Pipeline Parallelism

This module implements sequence partitioning algorithms and queue management for Seq1F1B 
pipeline parallelism. It provides dynamic sequence splitting based on computational balance 
and maintains efficient queue structures for managing sequence chunks during pipeline execution.

Key Components:
- Sequence1F1BQueue: Two-stage queue for managing sequence chunks in Seq1F1B schedule
- TFLOPs-based partitioning: Balances computational load across sequence splits
- Multiple splitting strategies: Average partitioning and TFLOPs-balanced partitioning
"""

import logging
import torch
from sympy import symbols, Eq, solve
from megatron.training import get_args
from .seq_utils import SeqTFlops

logger = logging.getLogger(__name__)


class Sequence1F1BQueue:
    def __init__(self, seq1f1b_splits=4, print_msg=False, add_msg=""):
        # two stage queue
        # first stage use offset to track the current queue
        # second stage use idx to track the current item
        self.queues = [[]]
        self.print_msg = print_msg
        self.info = add_msg
        self._offset = 0
        self._idx = 0
        self.count = 0
        self.seq1f1b_splits = seq1f1b_splits
        self.tail_obj = None

    def __getitem__(self, idx):
        self.print_log(f"get tail inp ")
        return self.tail_obj

    def __len__(self):
        return self.count

    def print_log(self, text, rank=0):
        if self.print_msg and torch.distributed.get_rank() == rank:
            logger.info(f"rank {rank} {self.info}: {text}")

    def append(self, obj):
        self.print_log("append inp")
        self.tail_obj = obj
        self.queues[self._offset].append(obj)
        self._idx += 1
        if self._idx == self.seq1f1b_splits:
            self.print_log("full queue , create new one")
            self.queues.append([])
            self._idx = 0
            self._offset += 1
        self.count += 1
    
    def pop(self, idx=0):
        self.print_log(f"pop head inp of first queue")
        self.count -= 1
        if len(self.queues[0]) == 1:
            if self._offset > 0:
                self._offset -= 1
                return self.queues.pop(idx)[0]
            else:
                return self.queues[idx].pop(-1)
        else:
            return self.queues[idx].pop(-1)


def get_tflops():
    args = get_args()
    config = {
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_hidden_size,
        "num_heads": args.num_attention_heads,
        "dim_head": args.hidden_size // args.num_attention_heads,
        "vocab_size": args.padded_vocab_size,
    }
    config = SeqTFlops(**config)
    tflops = config.get_seq_tflops(args.seq_length, causal=True)
    return tflops


def round_down(x, tp_size):
    return x // tp_size * tp_size


class solver:
    def __init__(self, total_seqlen, config, causal=True):
        self.total_seqlen = total_seqlen 
        self.config = config
        self.total_tflops = config.get_seq_tflops(total_seqlen, causal)
        

    def solve_partition(self, num_splits, tp_size=1):
        res = []
        prefix = self.total_seqlen
        for _ in range(1, num_splits):
            seqlen = symbols('seqlen')
            tflops = self.config.get_prefix_tflops(seqlen, prefix)
            eq = Eq(tflops, self.total_tflops / num_splits)
            sol = solve(eq, seqlen)
            sol = round_down(int(sol[0]), tp_size)
            res.insert(0, int(sol))
            prefix -= int(sol)
        res.insert(0, prefix)
        return res


partitions = None


def get_splits():
    global partitions
    args = get_args()
    if args.seq1f1b_balance_method == "average":
        return [args.seq_length // args.seq1f1b_splits] * args.seq1f1b_splits
    if args.seq1f1b_splits == 1:
        return [args.seq_length]
    if partitions is None:
        seqlen = args.seq_length
        config = {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "ffn_size": args.ffn_hidden_size,
            "num_heads": args.num_attention_heads,
            "dim_head": args.hidden_size // args.num_attention_heads,
            "vocab_size": args.padded_vocab_size,
        }
        tflops_config = SeqTFlops(**config)
        sol = solver(seqlen, tflops_config)
        args.total_tflops = sol.total_tflops
        mod = args.tensor_model_parallel_size if args.sequence_parallel else 1
        partitions = sol.solve_partition(args.seq1f1b_splits, mod)
    return partitions
