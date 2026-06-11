# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""
TTP Replica Optimizer - inherits DistributedOptimizer.

Key design:
1. self.data_parallel_group holds the incoming REPLICA sub-group.
2. gbuf_ranges are built on the REPLICA sub-group, each rank holds 1/2 shard.
3. self.ori_dp_group / ori_dp_list save the FULL DP group for dump index mapping.
4. Dump saves a 2-rank checkpoint; on recovery REPLICA sub-groups load independently.
5. TTP Processor is initialized on first optimizer creation (process-level idempotent).
6. step() is decorated with TTP exception handler + iteration tracking.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

logger = logging.getLogger(__name__)

# Process-level idempotent guards for TTP initialization
_TTP_INITIALIZED = False
_TTP_MAIN_THREAD_ID_SET = False

# Shared training-step counter — incremented by patched_chained_optimizer_step_wrapper
# in adaptor.py (once per ChainedOptimizer.step() call = once per training step),
# read by _ttp_on_step.  Matching mindio-ttp's get_iterations / set_iterations pattern.
_ttp_shared_step = 0
# Resolve the TTP exception handler at import time (no-op if unavailable)
try:
    from mindspeed.core.ttp.recovery.exception_handler import ttp_exception_handler
except ImportError:

    def ttp_exception_handler(func):
        return func


class TTPReplicaOptimizer(DistributedOptimizer):
    """
    TTP replica optimizer, inherits DistributedOptimizer.

    Key modifications:
    1. self.data_parallel_group holds the incoming REPLICA sub-group (not split).
    2. Save ori_dp_group / ori_dp_list for index mapping during dump.
    3. In dump mode, sharded_state_dict uses parent logic (validation skipped).
    4. On recovery, parallelization_group is replaced with the REPLICA sub-group.
    """

    def __init__(
        self,
        optimizer,
        config,
        grad_scaler,
        init_state_fn,
        model_chunks,
        per_model_buffers,
        data_parallel_group,
        data_parallel_group_gloo,
        data_parallel_group_idx,
        distributed_optimizer_instance_id,
        ori_dp_group=None,
        ori_dp_group_gloo=None,
    ):
        cur_rank = dist.get_rank()
        self.replica_num = 2
        self._original_distributed_optimizer_instance_id = None

        # === Save FULL DP group for dump index mapping ===
        # The incoming data_parallel_group is the FULL DP group (from verl / Megatron).
        # Save it BEFORE we swap to the REPLICA sub-group.
        if ori_dp_group is not None:
            self.ori_dp_group = ori_dp_group
            self.ori_dp_group_gloo = ori_dp_group_gloo
        else:
            self.ori_dp_group = data_parallel_group
            self.ori_dp_group_gloo = data_parallel_group_gloo
        self.ori_dp_list = list(dist.get_process_group_ranks(self.ori_dp_group))
        ori_dp_world_size = dist.get_world_size(self.ori_dp_group)

        if ori_dp_world_size == 1:
            raise ValueError("High availability does not support data_parallel_world_size=1")
        if ori_dp_world_size % self.replica_num != 0:
            raise ValueError(f"High availability does not support data_parallel_world_size({ori_dp_world_size})")

        # === Replace data_parallel_group with REPLICA sub-group ===
        # This mirrors the original patched_actor_init behavior: temporarily replace
        # _DATA_PARALLEL_GROUP with REPLICA before optimizer init. Now we do it here
        # by swapping the constructor parameter before passing to super().__init__().
        # Without this, gbuf_ranges would be built for the FULL DP (wrong size).
        from .replica_group import (
            ttp_initialize_replica_dp_group,
            get_dp_cp_replica_group,
            get_dp_cp_replica_group_gloo,
        )

        ttp_initialize_replica_dp_group()
        _replica_group = get_dp_cp_replica_group()
        _replica_group_gloo = get_dp_cp_replica_group_gloo()
        if _replica_group is not None:
            data_parallel_group = _replica_group
            data_parallel_group_gloo = _replica_group_gloo or data_parallel_group_gloo

        # === Swap buffer.data_parallel_group to REPLICA sub-group ===
        # _build_model_gbuf_range uses param_and_grad_buffer.data_parallel_group
        # to compute DP rank/world_size and split the gbuf.  We replace each
        # buffer's DP group with the REPLICA sub-group so that gbuf_ranges
        # are built for the replica DP world (DP=1 for our DP=2 topology),
        # giving each rank full coverage of its TP shard (1/4 of total).
        # After init we restore the original FULL DP group so that gradient
        # reduce-scatter still uses the correct number of ranks.
        _buf_original_dp_groups = []

        def _collect_buffers(buffers_container):
            """Recursively collect _ParamAndGradBuffer objects from any container type."""
            bufs = []
            if isinstance(buffers_container, dict):
                for _v in buffers_container.values():
                    bufs.extend(_collect_buffers(_v))
            elif isinstance(buffers_container, (list, tuple)):
                for _item in buffers_container:
                    bufs.extend(_collect_buffers(_item))
            elif hasattr(buffers_container, 'data_parallel_group'):
                bufs.append(buffers_container)
            return bufs

        for _b in _collect_buffers(per_model_buffers):
            _buf_original_dp_groups.append((_b, _b.data_parallel_group))
            _b.data_parallel_group = data_parallel_group

        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            model_chunks,
            per_model_buffers,
            data_parallel_group,
            data_parallel_group_gloo,
            data_parallel_group_idx,
            distributed_optimizer_instance_id,
        )

        # === Restore buffer.data_parallel_group to FULL DP ===
        for _b, _orig_group in _buf_original_dp_groups:
            _b.data_parallel_group = _orig_group

        # Fix distributed_optimizer_instance_id to use FULL DP position.
        # During init, data_parallel_group is swapped to REPLICA (size=1), so
        # super().__init__() sets instance_id=0 on every rank.  Use ori_dp_list
        # (FULL DP) position instead, so recovery-load DCP validation can
        # distinguish ranks within the same TP position.
        self.distributed_optimizer_instance_id = self.ori_dp_list.index(cur_rank)

        self.error_dump = False
        self.save_args = {}

        # === TTP one-time initialization (process-level idempotent) ===
        global _TTP_INITIALIZED, _TTP_MAIN_THREAD_ID_SET
        if not _TTP_INITIALIZED:
            _TTP_INITIALIZED = True
            logger.info("[TTP] TTP init on rank=%s", cur_rank)
            try:
                from mindspeed.core.ttp.recovery.dump_save import (
                    tft_init_controller_processor,
                    tft_register_processor,
                )

                tft_init_controller_processor()
                tft_register_processor()
                logger.info("[TTP] Worker initialization complete (rank=%s)", cur_rank)
            except Exception as e:
                logger.warning("[TTP] Initialization skipped (rank=%s): %s", cur_rank, e)

        if not _TTP_MAIN_THREAD_ID_SET:
            _TTP_MAIN_THREAD_ID_SET = True
            try:
                from mindspeed.core.ttp.comm.processor import TTPProcessor, set_worker_instance

                processor = TTPProcessor.get_instance()
                if processor:
                    processor.set_main_thread_id(threading.main_thread().ident)
                    set_worker_instance(self)  # self = TTPReplicaOptimizer, aka "worker" context
            except Exception as _e:
                logger.error(
                    "[TTP] Failed to set worker instance / main thread ID (rank=%s): %s",
                    cur_rank,
                    _e,
                    exc_info=True,
                )

    @ttp_exception_handler
    def step(self, *args, **kwargs):
        """Step with TTP iteration tracking + exception handling."""
        self._ttp_on_step()
        return super().step(*args, **kwargs)

    @ttp_exception_handler
    def step_with_ready_grads(self) -> bool:
        """Hook into the actual optimizer execution for fault injection.

        When TTPReplicaOptimizer is inside a ChainedOptimizer, the call chain
        is ChainedOptimizer.step() → step_with_ready_grads() on each sub-optimizer,
        which bypasses our step() override entirely.
        """
        self._ttp_on_step()
        return super().step_with_ready_grads()

    _TTP_LAST_STEP_TIME = 0.0
    _TTP_STEP_INTERVAL_THRESHOLD = 10.0  # > 1微批内 8 次调用的 ~7s 跨度, < 步间 ~60s 间隙

    @classmethod
    def _get_current_iteration(cls) -> int:
        """Time-based iteration counter — groups calls <5s apart as same step.

        ChainedOptimizer.step() is called once per micro-batch (multiple times
        per training step).  Time-based dedup ensures all micro-batch calls
        within a training step count as one iteration.
        """
        global _ttp_shared_step
        now = time.time()
        if now - cls._TTP_LAST_STEP_TIME > cls._TTP_STEP_INTERVAL_THRESHOLD:
            _ttp_shared_step += 1
            TTPReplicaOptimizer._TTP_LAST_STEP_TIME = now
        return _ttp_shared_step

    def _ttp_on_step(self):
        """Iteration tracking and fault injection, called on every optimizer step.

        Divides the raw counter by CALLS_PER_STEP (currently 8) to get the
        training step.  The divisor is determined empirically from the call
        pattern and is logged for verification.

        Diagnostic logging shows per-call topology info to help understand
        where the 8 calls/step comes from (TP rank, DP rank, etc.).
        """
        global _ttp_shared_step
        _ttp_shared_step += 1
        calls_per_step = 8  # empirically: 8 _ttp_on_step calls per training step
        iteration = _ttp_shared_step // calls_per_step

        try:
            from mindspeed.core.ttp.comm.processor import TTPProcessor

            processor = TTPProcessor.get_instance()
            if processor and not processor.is_inference_phase:
                processor.update_iteration(iteration)
        except Exception as _e:
            logger.warning("[TTP] update_iteration failed: %s", _e)

        # Fault injection is done in patched_forward_backward_wrapper (adaptor.py).

    @staticmethod
    def get_index_map(
        dp_ranks: List[int],
        sorted_save_rank_list: List[int],
        replica_num: int,
    ) -> Dict[int, int]:
        """Build a mapping from gather tensor index to FULL DP shard index.

        Used to reorder gathered optimizer state shards from REPLICA sub-group order
        back to FULL DP group order for checkpoint compatibility.
        """
        dp_size = len(dp_ranks)
        replica_size = dp_size // replica_num
        dp_ranks_tmp = [dp_ranks[i : i + replica_size] for i in range(0, dp_size, replica_size)]

        dp_ranks_maps = {}
        for data_parallel_ranks in dp_ranks_tmp:
            for i in range(replica_size):
                dp_ranks_maps[data_parallel_ranks[i]] = i

        tup = [(rank, si) for si, rank in enumerate(sorted_save_rank_list)]
        # Verify all ranks are in the map.  .get(rank, 0) would silently
        # group unknown ranks at position 0, corrupting shard ordering.
        _unknown = [r for r, _ in tup if r not in dp_ranks_maps]
        if _unknown:
            raise KeyError(
                f"Ranks {_unknown} not found in dp_ranks_maps. "
                f"sorted_save_rank_list={sorted_save_rank_list}, dp_ranks={dp_ranks}"
            )
        tup.sort(key=lambda x: dp_ranks_maps[x[0]])

        ti_to_si = {}
        for ti, (rank, si) in enumerate(tup):
            ti_to_si[ti] = si

        return ti_to_si

    def set_dump_args(self, rank: int, step: int, rank_list: List[int], global_rank: int = None) -> None:
        """Set dump parameters and switch the optimizer to dump mode.

        Recomputes the save_rank from ori_dp_list to ensure the first rank in
        each replica shard group is selected for writing the output file.
        """
        self.save_args['step'] = step
        self.save_args['rank'] = rank
        self.save_args['rank_list'] = rank_list
        self.error_dump = True

        if global_rank is not None:
            self.save_args['global_rank'] = global_rank
        else:
            self.save_args['global_rank'] = dist.get_rank()

        # Compute save_rank from ori_dp_list
        dp_size = len(self.ori_dp_list)
        replica_size = dp_size // self.replica_num
        dp_ranks_tmp = [self.ori_dp_list[i : i + replica_size] for i in range(0, dp_size, replica_size)]

        dp_ranks_maps = {}
        for data_parallel_ranks in dp_ranks_tmp:
            for i in range(replica_size):
                dp_ranks_maps[data_parallel_ranks[i]] = i

        for save_rank in rank_list:
            if dp_ranks_maps.get(save_rank) == 0:
                self.save_args['rank'] = save_rank
                break

        cur_rank = self.save_args['global_rank']
        rank_in_dump_group = rank_list.index(cur_rank) if cur_rank in rank_list else 0
        self._original_distributed_optimizer_instance_id = self.distributed_optimizer_instance_id
        self.distributed_optimizer_instance_id = rank_in_dump_group

    def need_write_file(self) -> bool:
        """Check if the current rank should write the optimizer state file."""
        cur_rank = dist.get_rank()
        return self.error_dump and self.save_args.get('rank') == cur_rank

    def get_parameter_state_dp_zero_for_ttp(self) -> Optional[dict]:
        """Gather and return the DP zero optimizer state for TTP dump.

        Only ranks in save_rank_list participate; the save_rank collects all shards.

        During dump, dump_save.py step 2.4 aligns each sub-optimizer's
        data_parallel_group with the dump ranks BEFORE this method is called.
        This guarantees gather covers all required ranks — no replica fill needed.
        """
        global_rank = self.save_args.get('global_rank', dist.get_rank())
        save_rank = self.save_args['rank']
        save_rank_list = self.save_args['rank_list']

        # After step 2.4 alignment, data_parallel_group and save_rank_list match.
        dp_world_size = dist.get_world_size(self.data_parallel_group)
        if dp_world_size != len(save_rank_list):
            raise ValueError(
                f"data_parallel size {dp_world_size} != save_rank_list size {len(save_rank_list)}. "
                f"data_parallel_group={list(dist.get_process_group_ranks(self.data_parallel_group))}, "
                f"save_rank_list={save_rank_list}"
            )

        gather_rank_list = sorted(save_rank_list)
        gather_world_size = len(gather_rank_list)

        # Build mapping from gather position to FULL DP shard position
        ti_to_si = self._build_gather_index_map(gather_rank_list, dp_world_size)

        save_group_gloo = dist.new_group(
            gather_rank_list,
            backend="gloo",
            use_local_synchronization=True,
        )
        if save_rank not in gather_rank_list:
            raise ValueError(f"save_rank({save_rank}) not in gather_rank_list({gather_rank_list})")
        save_rank_local = gather_rank_list.index(save_rank)

        return self.collect_param_state(
            global_rank,
            dp_world_size,
            gather_world_size,
            save_rank,
            save_rank_local,
            gather_rank_list,
            save_group_gloo,
            ti_to_si,
        )

    def _build_gather_index_map(self, gather_rank_list, dp_world_size):
        """
        Build gather index mapping.

        recv_tensors[ti] after gather corresponds to the shard of gather_rank_list[ti].
        ti_to_si[ti] = si means the shard of gather_rank_list[ti] should be placed at
        position si in the concatenated FULL DP result.

        With replica redundancy, the shard of rank i in the REPLICA sub-group
        corresponds to that rank's position within ori_dp_list in the FULL DP group.
        """
        rank_to_shard_idx = {rank: idx for idx, rank in enumerate(self.ori_dp_list)}
        # Verify all gather ranks exist in ori_dp_list — .get(rank, ti) would
        # silently map an unknown rank to its gather position, potentially
        # colliding with a legitimate mapping and overwriting a shard.
        _unknown_gather = set(gather_rank_list) - set(self.ori_dp_list)
        if _unknown_gather:
            raise KeyError(
                f"Gather ranks {_unknown_gather} not found in ori_dp_list "
                f"{self.ori_dp_list}. gather_rank_list={gather_rank_list}"
            )
        ti_to_si = {}
        for ti, rank in enumerate(gather_rank_list):
            ti_to_si[ti] = rank_to_shard_idx[rank]
        return ti_to_si

    def collect_param_state(
        self,
        global_rank: int,
        dp_world_size: int,
        gather_world_size: int,
        save_rank_global: int,
        save_rank_local: int,
        gather_rank_list: list,
        save_group_gloo: torch.distributed.ProcessGroup,
        ti_to_si: Dict[int, int],
    ) -> dict:
        """
        Collect and concatenate optimizer state.

        Args:
            dp_world_size: FULL DP group size, used to compute shard size (must match gbuf_ranges).
            gather_world_size: Number of ranks actually participating in gather (REPLICA sub-group size).
            save_rank_global: Rank responsible for collecting and saving (global rank, used for is_root check).
            save_rank_local: Rank responsible for collecting and saving (group-local rank, used for dist.gather dst).
            gather_rank_list: List of global ranks participating in the gather.
            save_group_gloo: gloo process group used for gather.
            ti_to_si: Mapping from gather position to FULL DP shard position.
        """
        state = {"buckets_coalesced": True}
        is_root = global_rank == save_rank_global

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            dtype_state = {}

            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded

                world_tensors = {}
                if is_root:
                    world_tensors = {
                        key: torch.zeros((buffer_numel_unpadded,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }
                    world_tensors["numel_unpadded"] = buffer_numel_unpadded

                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    offset_in_world_tensors = self._collect_bucket(
                        gbuf_idx,
                        bucket_idx,
                        gbuf_range_map,
                        dp_world_size,
                        gather_world_size,
                        save_rank_local,
                        save_group_gloo,
                        ti_to_si,
                        is_root,
                        world_tensors,
                        offset_in_world_tensors,
                    )

                dtype_state[dtype] = world_tensors
            state[gbuf_idx] = dtype_state

        return state

    def _collect_bucket(
        self,
        gbuf_idx,
        bucket_idx,
        gbuf_range_map,
        dp_world_size,
        gather_world_size,
        save_rank_local,
        save_group_gloo,
        ti_to_si,
        is_root,
        world_tensors,
        offset_in_world_tensors,
    ):
        """Collect and gather one bucket's optimizer state shards."""
        gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
        gbuf_local_numel = gbuf_world_numel // dp_world_size
        gbuf_world_numel_unpadded = self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded

        local_shards = {
            key: torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
            for key in ("param", "exp_avg", "exp_avg_sq")
        }

        for model_param, param_range_map in gbuf_range_map["param_map"].items():
            tensors = self._get_main_param_and_optimizer_states(model_param)
            gbuf_local_start = param_range_map["gbuf_local"].start
            gbuf_local_end = param_range_map["gbuf_local"].end
            for key in local_shards:
                local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(tensors[key].detach().cpu())

        for key, send_tensor in local_shards.items():
            # gather only collects shards from gather_world_size ranks
            if is_root:
                recv_tensors = [
                    torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                    for _ in range(gather_world_size)
                ]
            else:
                recv_tensors = []

            dist.gather(send_tensor, recv_tensors, save_rank_local, save_group_gloo)

            if is_root:
                # Arrange REPLICA sub-group shards in FULL DP order.
                full_dp_recv_tensors = [None] * dp_world_size
                for ti in range(gather_world_size):
                    si = ti_to_si[ti]
                    full_dp_recv_tensors[si] = recv_tensors[ti]

                unfilled = [i for i, t in enumerate(full_dp_recv_tensors) if t is None]
                if unfilled:
                    raise RuntimeError(
                        f"FULL DP positions {unfilled} unfilled after gather. "
                        f"dp_world_size={dp_world_size}, gather_world_size={gather_world_size}, "
                        f"ti_to_si={ti_to_si}"
                    )

                recv_tensors_concatenated = torch.cat(full_dp_recv_tensors)
                start = offset_in_world_tensors
                end = offset_in_world_tensors + gbuf_world_numel_unpadded
                world_tensors[key][start:end].copy_(recv_tensors_concatenated[:gbuf_world_numel_unpadded])

        offset_in_world_tensors += gbuf_world_numel_unpadded
        return offset_in_world_tensors

    def save_parameters_state_ttp(self) -> Optional[dict]:
        """Collect param state in dump mode. Only the save_rank returns the state dict."""
        cur_rank = self.save_args.get('global_rank', dist.get_rank())
        save_rank = self.save_args['rank']

        if cur_rank not in self.save_args['rank_list']:
            return None

        state_dict = self.get_parameter_state_dp_zero_for_ttp()
        if cur_rank == save_rank:
            return state_dict
        return None

    def save_parameter_state_impl(self) -> Optional[dict]:
        """Route to dump-mode or normal-mode parameter state collection."""
        if self.error_dump:
            return self.save_parameters_state_ttp()
        return self.get_parameter_state_dp_zero()

    def save_parameter_state(self, filename: str) -> None:
        """Save parameter state to file. In dump mode, only the designated rank writes."""
        state_dict = self.save_parameter_state_impl()

        if self.error_dump:
            save_rank = self.save_args['rank']
            cur_rank = self.save_args.get('global_rank', dist.get_rank())
            if cur_rank == save_rank and state_dict is not None:
                torch.save(state_dict, filename)
        else:
            if dist.get_rank(self.ori_dp_group) == 0:
                torch.save(state_dict, filename)

    def sharded_state_dict(
        self,
        model_sharded_state_dict,
        is_loading: bool = False,
        sharding_type: str = 'fully_sharded_model_space',
        **kwargs,
    ):
        import megatron.core
        from packaging import version

        _mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
        if _mcore_ge_014:
            # Megatron >= 0.14.0 accepts sharding_type and metadata in
            # DistributedOptimizer.sharded_state_dict().  Forward them.
            return super().sharded_state_dict(
                model_sharded_state_dict,
                is_loading=is_loading,
                sharding_type=sharding_type,
                **kwargs,
            )
        else:
            # Megatron < 0.14.0 only accepts (self, model_sharded_state_dict,
            # is_loading=False) — no sharding_type, no kwargs.
            return super().sharded_state_dict(
                model_sharded_state_dict,
                is_loading=is_loading,
            )
