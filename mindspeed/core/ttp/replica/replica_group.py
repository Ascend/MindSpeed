# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Replica group management for TTP high availability.

Provides global state for DP/CP/EP replica groups and utilities to build
replica sub-groups from Megatron parallel state.
"""

import logging
import threading
from typing import List, Optional, Dict, Any

import torch.distributed as dist

from ..constants import WorkerStatus

logger = logging.getLogger(__name__)

DUMP_WORLD_GROUP = None
DP_CP_ORIGIN_RANKS: Optional[List[int]] = None
DP_ORIGIN_RANKS: Optional[List[int]] = None
DP_EP_ORIGIN_RANKS: Optional[List[int]] = None
DP_CP_REPLICA_GROUP = None
DP_CP_REPLICA_GROUP_GLOO = None
GLOBAL_DP_CP_RANKS: Optional[List[List[int]]] = None
GLOBAL_DP_EP_RANKS: Optional[List[List[int]]] = None
REPLICA_NUM = 2


def set_dump_world_group(group):
    global DUMP_WORLD_GROUP
    DUMP_WORLD_GROUP = group


def get_dump_world_group():
    return DUMP_WORLD_GROUP


def reset_dp_cp_replica_group(group, group_gloo=None):
    global DP_CP_REPLICA_GROUP, DP_CP_REPLICA_GROUP_GLOO
    DP_CP_REPLICA_GROUP = group
    DP_CP_REPLICA_GROUP_GLOO = group_gloo


def get_dp_cp_replica_group():
    return DP_CP_REPLICA_GROUP


def get_dp_cp_replica_group_gloo():
    return DP_CP_REPLICA_GROUP_GLOO


def set_global_dp_cp_ranks(ranks: List[List[int]]) -> None:
    global GLOBAL_DP_CP_RANKS
    GLOBAL_DP_CP_RANKS = ranks


def get_global_dp_cp_ranks() -> List[List[int]]:
    return GLOBAL_DP_CP_RANKS or []


def set_global_dp_ep_ranks(ranks: List[List[int]]) -> None:
    global GLOBAL_DP_EP_RANKS
    GLOBAL_DP_EP_RANKS = ranks


def get_global_dp_ep_ranks() -> List[List[int]]:
    return GLOBAL_DP_EP_RANKS or []


def get_replica_num() -> int:
    return REPLICA_NUM


def get_dp_cp_ranks() -> List[int]:
    return DP_CP_ORIGIN_RANKS or []


def get_dp_ranks() -> List[int]:
    return DP_ORIGIN_RANKS or []


def ttp_get_replica_dp_num() -> int:
    if GLOBAL_DP_CP_RANKS is None or len(GLOBAL_DP_CP_RANKS) == 0:
        return REPLICA_NUM

    dp_size = len(GLOBAL_DP_CP_RANKS[0]) if GLOBAL_DP_CP_RANKS else 1
    return max(1, dp_size // REPLICA_NUM)


def build_dp_cp_replica_group(dp_cp_ranks: list, is_first: bool):
    """Build REPLICA sub-groups for DP+CP ranks and create NCCL/GLOO process groups."""
    global DP_CP_REPLICA_GROUP, DP_CP_REPLICA_GROUP_GLOO

    if len(dp_cp_ranks) % REPLICA_NUM != 0:
        raise ValueError(f"High availability do not support the size of dp_cp_ranks:{dp_cp_ranks} ")

    cur_rank = dist.get_rank()
    replica_group_size = len(dp_cp_ranks) // REPLICA_NUM
    # Divide dp_cp_ranks into replica sub-lists by REPLICA_NUM
    replica_lists = [dp_cp_ranks[i * replica_group_size : (i + 1) * replica_group_size] for i in range(REPLICA_NUM)]

    for replica_list in replica_lists:
        if is_first:
            # Create nccl and gloo process groups for each replica sub-list
            replica_group = dist.new_group(replica_list, use_local_synchronization=True)
            replica_group_gloo = dist.new_group(replica_list, backend="gloo", use_local_synchronization=False)
            # Only set global replica group vars if current rank belongs to this replica
            if cur_rank in replica_list:
                DP_CP_REPLICA_GROUP = replica_group
                DP_CP_REPLICA_GROUP_GLOO = replica_group_gloo


def build_dp_ep_replica_group(dp_ep_ranks: list, is_first: bool):
    """Build REPLICA sub-groups for DP+EP ranks and create NCCL/GLOO process groups."""
    if len(dp_ep_ranks) % REPLICA_NUM != 0:
        raise ValueError(f"High availability do not support the size of dp_ep_ranks:{dp_ep_ranks} ")

    replica_group_size = len(dp_ep_ranks) // REPLICA_NUM
    replica_lists = [dp_ep_ranks[i * replica_group_size : (i + 1) * replica_group_size] for i in range(REPLICA_NUM)]

    for replica_list in replica_lists:
        if is_first:
            # Create process groups for NCCL/GLOO communication
            dist.new_group(replica_list, use_local_synchronization=True)
            dist.new_group(replica_list, backend="gloo", use_local_synchronization=False)


def ttp_initialize_replica_dp_group():
    """Initialize replica DP groups from Megatron parallel state topology."""
    global DP_CP_ORIGIN_RANKS, DP_ORIGIN_RANKS, DP_EP_ORIGIN_RANKS
    global GLOBAL_DP_CP_RANKS, GLOBAL_DP_EP_RANKS

    if DP_CP_REPLICA_GROUP is not None:
        return

    from megatron.core import parallel_state
    from megatron.core.parallel_state import RankGenerator

    cur_rank = dist.get_rank()
    world_size = dist.get_world_size()

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    cp_size = getattr(parallel_state, 'get_context_parallel_world_size', lambda: 1)()
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    etp_size = getattr(parallel_state, 'get_expert_tensor_parallel_world_size', lambda: tp_size)

    decoder_model_size = tp_size * pp_size * cp_size
    dp_size = world_size // decoder_model_size
    order = "tp-cp-ep-dp-pp"

    # Decoder: iterate all DP+CP groups and build replica sub-groups for each
    rank_gen = RankGenerator(tp=tp_size, ep=1, dp=dp_size, pp=pp_size, cp=cp_size, order=order)

    GLOBAL_DP_CP_RANKS = []
    for dp_cp_ranks in rank_gen.get_ranks('dp-cp'):
        dp_cp_ranks = list(dp_cp_ranks)
        if cur_rank in dp_cp_ranks:
            DP_CP_ORIGIN_RANKS = dp_cp_ranks
        build_dp_cp_replica_group(dp_cp_ranks, is_first=True)
        GLOBAL_DP_CP_RANKS.append(dp_cp_ranks)

    # DP ranks (excluding CP)
    for dp_ranks in rank_gen.get_ranks('dp'):
        dp_ranks = list(dp_ranks)
        if cur_rank in dp_ranks:
            DP_ORIGIN_RANKS = dp_ranks

    # Expert: iterate all DP+EP groups for MoE scenarios
    if ep_size > 1:
        expert_dp_size = world_size // (etp_size * ep_size * pp_size)
        expert_rank_gen = RankGenerator(tp=etp_size, ep=ep_size, dp=expert_dp_size, pp=pp_size, cp=1, order=order)

        GLOBAL_DP_EP_RANKS = []
        for dp_ep_ranks in expert_rank_gen.get_ranks('dp'):
            dp_ep_ranks = list(dp_ep_ranks)
            if cur_rank in dp_ep_ranks:
                DP_EP_ORIGIN_RANKS = dp_ep_ranks
            build_dp_ep_replica_group(dp_ep_ranks, is_first=True)
            GLOBAL_DP_EP_RANKS.append(dp_ep_ranks)


def tft_reset_dp_cp_replica_group(group):
    """Reset REPLICA sub-group (used by dump_save.py)."""
    return reset_dp_cp_replica_group(group)


class ReplicaGroupManager:
    """Builds and manages DP/CP and DP/EP replica groups from Megatron parallel state."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.dp_cp_groups: List[List[int]] = []
        self.dp_ep_groups: List[List[int]] = []
        self.dp_cp_rep_cnt: int = 2
        self.dp_ep_rep_cnt: int = 2
        self.dp_size = 1
        self.tp_size = 1
        self.ep_size = 1
        self.cp_size = 1

    def build_groups_from_megatron(self) -> None:
        try:
            from megatron.core import parallel_state

            self.dp_size = parallel_state.get_data_parallel_world_size()
            self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
            self.ep_size = parallel_state.get_expert_model_parallel_world_size()
            self.cp_size = getattr(parallel_state, 'get_context_parallel_world_size', lambda: 1)()

            dp_cp_groups = self._build_dp_cp_groups()
            dp_ep_groups = self._build_dp_ep_groups()

            self.dp_cp_groups = dp_cp_groups
            self.dp_ep_groups = dp_ep_groups
            self.dp_cp_rep_cnt = get_replica_num()
            self.dp_ep_rep_cnt = get_replica_num()

            set_global_dp_cp_ranks(dp_cp_groups)
            set_global_dp_ep_ranks(dp_ep_groups)

            logger.info(
                "Built replica groups: dp_size=%s, tp_size=%s, ep_size=%s, cp_size=%s",
                self.dp_size,
                self.tp_size,
                self.ep_size,
                self.cp_size,
            )
            logger.info("DP_CP groups: %s", dp_cp_groups)
            logger.info("DP_EP groups: %s", dp_ep_groups)

        except ImportError:
            logger.warning("Megatron not available, using default replica groups")
            self._build_default_groups()
        except Exception as e:
            logger.warning("Failed to build groups from Megatron: %s, using default groups", e)
            self._build_default_groups()

    def _build_dp_cp_groups(self) -> List[List[int]]:
        try:
            from megatron.core import parallel_state

            dp_cp_groups = []
            dp_world_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
            dp_world_ranks = parallel_state.get_data_parallel_group_gloo(with_context_parallel=True)

            if dp_world_ranks is not None:
                all_ranks = list(dist.get_process_group_ranks(dp_world_ranks))
            else:
                all_ranks = list(range(dp_world_size))

            logger.info("Building DP_CP groups: all_ranks=%s, dp_world_size=%s", all_ranks, dp_world_size)

            if not all_ranks:
                logger.warning("all_ranks is empty, using fallback: all ranks in one group")
                return [list(range(self.world_size))]

            # Return the full DP group as-is — do NOT split into replicas here.
            # Replica splitting is done by choose_rank_inner_rl in the Controller
            # using rep_cnt.  Each worker only sees its local DP group (e.g. [0,4]
            # for TP=4,DP=2), so the Controller merges groups from all workers.
            dp_cp_groups = [all_ranks]

            logger.info("Built DP_CP groups: %s", dp_cp_groups)
            return dp_cp_groups

        except Exception as e:
            logger.warning("Failed to build DP_CP groups: %s", e)
            return [list(range(self.world_size))]

    def _build_dp_ep_groups(self) -> List[List[int]]:
        try:
            from megatron.core import parallel_state

            if self.ep_size <= 1:
                return []

            dp_ep_groups = []
            dp_world_size = parallel_state.get_data_parallel_world_size(with_expert_parallel=True)
            dp_world_ranks = parallel_state.get_data_parallel_group_gloo(with_expert_parallel=True)

            if dp_world_ranks is not None:
                all_ranks = list(dist.get_process_group_ranks(dp_world_ranks))
            else:
                all_ranks = list(range(dp_world_size))

            # Return full DP group — replica splitting is done by the Controller
            if all_ranks:
                dp_ep_groups = [all_ranks]

            return dp_ep_groups

        except Exception as e:
            logger.warning("Failed to build DP_EP groups: %s", e)
            return []

    def _build_default_groups(self) -> None:
        all_ranks = list(range(self.world_size))
        replica_num = get_replica_num()
        group_size = self.world_size // replica_num if replica_num > 0 else self.world_size

        dp_cp_groups = []
        for i in range(replica_num):
            group_ranks = all_ranks[i * group_size : (i + 1) * group_size]
            if group_ranks:
                dp_cp_groups.append(group_ranks)

        self.dp_cp_groups = dp_cp_groups
        self.dp_ep_groups = []

        set_global_dp_cp_ranks(dp_cp_groups)
        set_global_dp_ep_ranks([])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rank': self.rank,
            'world_size': self.world_size,
            'dp_cp_groups': self.dp_cp_groups,
            'dp_ep_groups': self.dp_ep_groups,
            'dp_cp_rep_cnt': self.dp_cp_rep_cnt,
            'dp_ep_rep_cnt': self.dp_ep_rep_cnt,
            'dp_size': self.dp_size,
            'tp_size': self.tp_size,
            'ep_size': self.ep_size,
            'cp_size': self.cp_size,
        }


class ServerReplicaGroupManager:
    """Aggregates replica group info from all workers and selects dump ranks on fault."""

    def __init__(self):
        self.worker_replica_info: Dict[int, Dict[str, Any]] = {}
        self.dp_cp_groups: List[List[int]] = []
        self.dp_ep_groups: List[List[int]] = []
        self.dp_cp_rep_cnt: int = 2
        self.dp_ep_rep_cnt: int = 2
        self.worker_status: Dict[int, Any] = {}
        self._lock = threading.Lock()

    def update_from_worker(self, rank: int, data: Dict[str, Any]) -> None:
        with self._lock:
            self.worker_replica_info[rank] = data

            # Merge dp_cp_groups from all workers (each sends its own DP group).
            # Every worker sees only its local DP group (e.g. [0,4] for TP=4, DP=2),
            # so we must collect groups from ALL workers to build the full topology.
            if 'dp_cp_groups' in data and data['dp_cp_groups']:
                for group in data['dp_cp_groups']:
                    sorted_group = sorted(group)
                    if sorted_group not in self.dp_cp_groups:
                        self.dp_cp_groups.append(sorted_group)
            if 'dp_ep_groups' in data and data['dp_ep_groups']:
                for group in data['dp_ep_groups']:
                    sorted_group = sorted(group)
                    if sorted_group not in self.dp_ep_groups:
                        self.dp_ep_groups.append(sorted_group)
            if 'dp_cp_rep_cnt' in data:
                self.dp_cp_rep_cnt = data['dp_cp_rep_cnt']
            if 'dp_ep_rep_cnt' in data:
                self.dp_ep_rep_cnt = data['dp_ep_rep_cnt']

    def update_worker_status(self, rank: int, status) -> None:
        with self._lock:
            self.worker_status[rank] = status

    def select_dump_ranks(self, fault_ranks: List[int]) -> List[int]:
        """Select a healthy replica sub-group (excluding fault_ranks) for dump."""
        with self._lock:
            dump_ranks = []

            logger.info("Selecting dump ranks for fault_ranks=%s", fault_ranks)
            logger.info("Current dp_cp_groups=%s, worker_status=%s", self.dp_cp_groups, self.worker_status)

            if self.dp_cp_groups:
                for dp_group in self.dp_cp_groups:
                    if not dp_group:
                        continue

                    replica_num = get_replica_num()
                    if replica_num <= 0:
                        replica_num = 1

                    if replica_num > 1:
                        rank_size = len(dp_group)
                        offset = rank_size // replica_num

                        logger.info(
                            "ChooseRankInnerRL: rank_size=%s, replica_num=%s, offset=%s", rank_size, replica_num, offset
                        )

                        replica_lists = []
                        for i in range(replica_num):
                            replica_list = dp_group[i * offset : (i + 1) * offset]
                            replica_lists.append(replica_list)

                        logger.info("Replica lists: %s", replica_lists)

                        for replica_list in replica_lists:
                            has_fault = any(r in fault_ranks for r in replica_list)

                            if has_fault:
                                logger.info("Replica %s has fault, skipping", replica_list)
                                continue

                            all_healthy = all(self.worker_status.get(r) != WorkerStatus.FAULT for r in replica_list)

                            if all_healthy:
                                logger.info("Found healthy replica: %s", replica_list)
                                dump_ranks = replica_list
                                break
                            unhealthy_ranks = [
                                r for r in replica_list if self.worker_status.get(r) == WorkerStatus.FAULT
                            ]
                            logger.info("Replica %s has unhealthy ranks: %s, skipping", replica_list, unhealthy_ranks)

                        if dump_ranks:
                            break
                    else:
                        healthy_ranks = [
                            r
                            for r in dp_group
                            if r not in fault_ranks and self.worker_status.get(r) != WorkerStatus.FAULT
                        ]
                        if healthy_ranks:
                            dump_ranks = healthy_ranks
                            break
            else:
                pass

            if not dump_ranks and self.dp_ep_groups:
                for ep_group in self.dp_ep_groups:
                    if not ep_group:
                        continue

                    replica_num = get_replica_num()
                    if replica_num <= 0:
                        replica_num = 1

                    if replica_num > 1:
                        rank_size = len(ep_group)
                        offset = rank_size // replica_num

                        replica_lists = []
                        for i in range(replica_num):
                            replica_list = ep_group[i * offset : (i + 1) * offset]
                            replica_lists.append(replica_list)

                        for replica_list in replica_lists:
                            has_fault = any(r in fault_ranks for r in replica_list)
                            if has_fault:
                                continue

                            all_healthy = all(self.worker_status.get(r) != WorkerStatus.FAULT for r in replica_list)

                            if all_healthy:
                                dump_ranks = replica_list
                                break

                        if dump_ranks:
                            break
                    else:
                        healthy_ranks = [
                            r
                            for r in ep_group
                            if r not in fault_ranks and self.worker_status.get(r) != WorkerStatus.FAULT
                        ]
                        if healthy_ranks:
                            dump_ranks = healthy_ranks
                            break

            if not dump_ranks:
                logger.info("No dump ranks from groups, trying worker_replica_info")
                for rank, info in self.worker_replica_info.items():
                    if rank not in fault_ranks:
                        if self.worker_status.get(rank) != WorkerStatus.FAULT:
                            dump_ranks.append(rank)
                            break

            if not dump_ranks:
                logger.warning(
                    "No healthy dump ranks found! fault_ranks=%s, dp_cp_groups=%s, worker_status=%s",
                    fault_ranks,
                    self.dp_cp_groups,
                    self.worker_status,
                )

            result = list(set(dump_ranks))
            logger.info("Selected dump ranks: %s", result)
            return result
