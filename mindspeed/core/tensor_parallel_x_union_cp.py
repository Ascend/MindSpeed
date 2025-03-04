# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.distributed

from megatron.core.parallel_state import get_nccl_options

from mindspeed.core.mindspeed_parallel_group import MindspeedParallelGroup
from mindspeed.core.simple_parallel_cfg import SimpleParallelCfg
from mindspeed.core.singleton_meta import SingletonMeta


class TensorParallelXUnionCP(MindspeedParallelGroup, metaclass=SingletonMeta):
    def __init__(
        self,
        parallel_cfg: SimpleParallelCfg = None,
        pg_name: str = None,
        overlap_gp_name: str = None,
        nccl_comm_cfgs=None,
    ):
        super().__init__(parallel_cfg, pg_name, overlap_gp_name, nccl_comm_cfgs)

    @staticmethod
    def init_group(
        parallel_cfg: SimpleParallelCfg,
        pg_name: str,
        overlap_gp_name: str = None,
        nccl_comm_cfgs=None,
    ):
        pp = parallel_cfg.pp
        tp = parallel_cfg.tp
        cp = parallel_cfg.cp
        tp_x = parallel_cfg.tp_x

        rank = torch.distributed.get_rank()
        world_size: int = torch.distributed.get_world_size()
        num_pp_groups: int = world_size // pp
        dp = world_size // (tp * pp * cp)

        all_cp_grps = []
        for i in range(pp):
            for j in range(dp):
                start_rank = i * num_pp_groups + j * tp * cp
                end_rank = i * num_pp_groups + (j + 1) * tp * cp
                for k in range(tp):
                    ranks = range(start_rank + k, end_rank, tp)
                    all_cp_grps.append(list(ranks))

        all_tp_x_grps = []
        num_tp_grps: int = world_size // tp
        for i in range(num_tp_grps):
            for j in range(tp // tp_x):
                ranks = range(i * tp + j * tp_x, i * tp + (j + 1) * tp_x)
                all_tp_x_grps.append(list(ranks))

        # Build the tensor model-parallel-x-cp groups.
        res_group, res_overlap_group, res_global_ranks = None, None, None
        tp_x_cp_grp_ranks = []
        tp_x_cp_rank_in_grp = []
        for cp_grp in all_cp_grps:
            for cp_rank in cp_grp:
                for tp_x_grp in all_tp_x_grps:
                    if cp_rank in tp_x_grp and tp_x_grp not in tp_x_cp_rank_in_grp:
                        tp_x_cp_rank_in_grp += tp_x_grp

            tp_x_cp_rank_in_grp = sorted(tp_x_cp_rank_in_grp)
            if tp_x_cp_rank_in_grp not in tp_x_cp_grp_ranks:
                tp_x_cp_grp_ranks.append(tp_x_cp_rank_in_grp)
            tp_x_cp_rank_in_grp = []

        cur_overlap_group = None
        for tp_x_cp_ranks in tp_x_cp_grp_ranks:
            cur_group = torch.distributed.new_group(
                tp_x_cp_ranks, pg_options=get_nccl_options(pg_name, nccl_comm_cfgs)
            )
            # for send-recv in parallel
            if overlap_gp_name:
                cur_overlap_group = torch.distributed.new_group(
                    tp_x_cp_ranks, pg_options=get_nccl_options(overlap_gp_name, nccl_comm_cfgs),
                )

            if rank in tp_x_cp_ranks:
                res_global_ranks = tp_x_cp_ranks
                res_group = cur_group
                res_overlap_group = cur_overlap_group

        return res_group, res_global_ranks, res_overlap_group
