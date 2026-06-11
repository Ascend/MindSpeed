# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""TTP adaptor: Megatron-layer patch function definitions (pure library).

Each function is a module-level patch implementation. They are consumed by
features_manager/ttp/dump_patches.py, load_patches.py, and init.py.

All functions use the `_wrapper` naming convention (even for full replacements)
so that Patch.apply_patch / Patch.remove_patch can be called repeatedly.
"""

import functools
import logging
import os

logger = logging.getLogger(__name__)


# ==============================================================================
# Fault injection (inlined — no external file dependency)
# ==============================================================================

_TTP_FAULT_INJECTED = False


def _ttp_inject_fault_if_enabled(iteration):
    """Inject fault when env vars enable it and current iteration/rank match.

    Controlled by env vars:
      TTP_FAULT_INJECTION_ENABLED  — 'true' to enable (default: false)
      TTP_FAULT_INJECTION_ITERATION — target iteration (default: 3)
      TTP_FAULT_INJECTION_RANK     — target rank (default: 1)

    Has a once-only guard: after the first injection, subsequent calls
    are no-ops.
    """
    global _TTP_FAULT_INJECTED
    if _TTP_FAULT_INJECTED:
        return

    enabled = os.environ.get('TTP_FAULT_INJECTION_ENABLED', 'false').lower() == 'true'
    if not enabled:
        return

    target_iteration = int(os.environ.get('TTP_FAULT_INJECTION_ITERATION', '3'))
    target_rank = int(os.environ.get('TTP_FAULT_INJECTION_RANK', '1'))

    if iteration != target_iteration:
        return

    try:
        import torch.distributed as dist

        current_rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK', '0'))
    except Exception:
        current_rank = int(os.environ.get('RANK', '0'))

    if current_rank != target_rank:
        return

    _TTP_FAULT_INJECTED = True
    raise RuntimeError(f"TTP_INJECTED_FAULT at iteration {iteration} on rank {target_rank}")


# ==============================================================================
# Permanent patch
# ==============================================================================


def patched_optimizer_new(cls, *args, **kwargs):
    """Replace DistributedOptimizer.__new__ to return TTPReplicaOptimizer.

    When TTP is enabled and a REPLICA sub-group exists, constructing
    DistributedOptimizer returns a TTPReplicaOptimizer instead.
    Also triggers idempotent REPLICA sub-group construction.
    """
    # Lazy imports — only resolved at call time (optimizer construction).
    try:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        return object.__new__(cls)

    if cls is not DistributedOptimizer:
        return object.__new__(cls)

    try:
        from mindspeed.core.ttp.replica.replica_group import (
            ttp_initialize_replica_dp_group,
            get_dp_cp_replica_group,
        )

        ttp_initialize_replica_dp_group()
        replica_group = get_dp_cp_replica_group()
        if replica_group is not None:
            from mindspeed.core.ttp.replica.replica_optimizer import TTPReplicaOptimizer

            return object.__new__(TTPReplicaOptimizer)
    except Exception as e:
        logger.warning("[TTP] optimizer __new__ fallback to DistributedOptimizer: %s", e)
    return object.__new__(cls)


_ORIGINAL_CHAINED_OPTIMIZER_INIT = None


def patched_chained_optimizer_init(self, *args, **kwargs):
    """Patch ChainedOptimizer.__init__ to store parent reference on TTP sub-optimizers.

    When TTP is enabled, each DistributedOptimizer inside ChainedOptimizer is replaced
    with TTPReplicaOptimizer.  The save callback needs the FULL ChainedOptimizer
    (not a sub-optimizer) to produce a complete sharded_state_dict.  Storing
    _ttp_parent_chained on each TTPReplicaOptimizer lets _get_actor_optimizer_from_worker
    return the parent.
    """
    _ORIGINAL_CHAINED_OPTIMIZER_INIT(self, *args, **kwargs)  # pylint: disable=not-callable

    for opt in self.chained_optimizers:
        if hasattr(opt, 'ori_dp_group'):
            opt._ttp_parent_chained = self


# ==============================================================================
# Permanent patch: forward_backward schedule PAUSE check + fault injection
# ==============================================================================

# Iteration for fault injection comes from _ttp_shared_step in replica_optimizer,
# NOT from a forward_backward counter.  forward_backward_no_pipelining is called
# for ref_log_prob / old_log_prob inference too, so counting those calls inflates
# the counter and triggers the fault too early.  _ttp_shared_step is incremented
# only during optimizer steps (actual training).
#
# We compute the NEXT expected training step: (_ttp_shared_step // CALLS_PER_STEP) + 1.
# Example: after step 4 completes (_ttp_shared_step=32), next_step=5 → fault fires.


def patched_forward_backward_wrapper(original_func):
    """Patch Megatron's forward_backward schedule to check PAUSE.

    Without this, PAUSE injected during the forward pass goes undetected until
    the next optimizer step — because @ttp_exception_handler only wraps
    step_with_ready_grads().  The async exception (PyThreadState_SetAsyncExc)
    may also be delayed in C calls.

    Fault injection has been moved to patched_chained_optimizer_step_wrapper
    (post-optimizer-step) to align with verl's normal checkpoint save point.
    verl saves at the end of each training step (after optimizer completes),
    naming the checkpoint global_step_N with post-step-N weights.  Injecting
    at forward_backward entry produced a dump named global_step_N with
    post-step-(N-1) weights — one step off.
    """

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        from mindspeed.core.ttp.comm.processor import TTPProcessor
        from mindspeed.core.ttp.recovery.exception_handler import _handle_pause_and_dump

        processor = TTPProcessor.get_instance()

        try:
            if processor:
                processor.check_and_raise_if_paused()

            return original_func(*args, **kwargs)
        except RuntimeError as e:
            error_str = str(e)

            # STEP FINISH — main thread was PAUSEd by Controller
            is_step_finish = error_str == "STEP FINISH" or (processor is not None and processor._pause_type == 'RAISE')
            if is_step_finish and processor is not None:
                _handle_pause_and_dump(processor)

            # TTP_INJECTED_FAULT — intentional fault injection, report and pause
            if "TTP_INJECTED_FAULT" in error_str and processor is not None:
                processor.on_worker_exception(e)
                _handle_pause_and_dump(processor)

            raise

    return wrapper


def patched_chained_optimizer_step_wrapper(original_func):
    """Wrap ChainedOptimizer.step() with TTP exception handler and fault injection.

    The ChainedOptimizer.step() patched by count_zero_fix calls
    clip_grad_by_total_norm_fp32 BEFORE step_with_ready_grads().
    If stop_device fires during gradient clipping, the FORCE STOP
    exception is NOT caught by @ttp_exception_handler (which only
    wraps step_with_ready_grads and forward_backward).  This wrapper
    adds exception handling at the ChainedOptimizer.step level so
    that FORCE STOP at any point during the entire optimizer step
    (including clip_grads) is caught and routed to _handle_pause_and_dump.

    Fault injection happens AFTER the optimizer step completes.  This
    aligns with verl's normal checkpoint save point — verl saves
    after _update_actor returns (post-optimizer), naming the checkpoint
    global_step_N with post-step-N weights.  Injecting here means our
    dump also contains post-step-N weights, matching verl's convention.
    """
    from mindspeed.core.ttp.recovery.exception_handler import ttp_exception_handler

    @functools.wraps(original_func)
    @ttp_exception_handler
    def wrapper(self, *args, **kwargs):
        result = original_func(self, *args, **kwargs)

        # --- fault injection AFTER optimizer step completes ---
        # Uses the optimizer step counter directly: _ttp_shared_step // 8
        # gives the CURRENT completed iteration.  Example: after all 8
        # micro-batches of step 11, _ttp_shared_step=88, iteration=11.
        # This matches verl's convention: global_step_11 = post-step-11 weights.
        try:
            from mindspeed.core.ttp.replica.replica_optimizer import _ttp_shared_step as _opt_step

            _iteration = _opt_step // 8
        except Exception:
            _iteration = 0
        if _iteration > 0:
            _ttp_inject_fault_if_enabled(_iteration)

        return result

    return wrapper


# ==============================================================================
# Dump patch functions
# ==============================================================================


def patched_save_preprocess_wrapper(original_func):
    """Force validate_access_integrity=False during dump/load."""

    @functools.wraps(original_func)
    def wrapper(sharded_state_dict, validate_access_integrity=True, preprocess_common_before_consistancy_check=None):
        return original_func(
            sharded_state_dict,
            validate_access_integrity=False,
            preprocess_common_before_consistancy_check=preprocess_common_before_consistancy_check,
        )

    return wrapper


def patched_apply_saving_parallelization_wrapper(original_func):
    """Skip validate_sharding_integrity in FullyParallelSaveStrategyWrapper.

    Reimplements apply_saving_parallelization without calling
    validate_sharding_integrity or determine_global_metadata.
    Compatible with the proven implementation in save_callback.py section 4.7.
    """

    @functools.wraps(original_func)
    def wrapper(self, sharded_state_dict):
        import time

        from megatron.core.dist_checkpointing.strategies.fully_parallel import (
            logger as fp_logger,
            determine_main_replica_uniform_distribution,
            distribute_main_replicas_with_precomputed_distribution,
        )

        start = time.time()
        if self.do_cache_distribution and self.cached_distribution is not None:
            fp_logger.debug('Apply *cached* save parallelization')
            precomputed_distribution = self.cached_distribution
        else:
            fp_logger.debug('Apply save parallelization')
            # ignore_groups=True: include ALL ShardedTensors regardless of
            # their replica_id.  Our dump ranks (4,5,6,7) have dp_rank=1
            # (replica_id=1), which is_main_replica() rejects.  Without
            # ignore_groups=True, only a handful of optimizer tensors that
            # happen to have replica_id=0 are saved — model weights and
            # most optimizer states are silently dropped.
            precomputed_distribution = determine_main_replica_uniform_distribution(
                sharded_state_dict, self.parallelization_group, ignore_groups=True
            )

        from megatron.core.dist_checkpointing.mapping import ShardedObject as _ShardedObject
        from megatron.core.dist_checkpointing.dict_utils import nested_values as _nested_values

        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, precomputed_distribution
        )

        # distribute_main_replicas_with_precomputed_distribution only handles
        # ShardedTensor (line 445 in fully_parallel.py).  ShardedObjects
        # (_extra_state, rng_state, optimizer base state) keep their original
        # replica_id (1 on dump ranks).  Set them to 0 so is_main_replica()
        # recognizes them and they are included in the save.
        for sh_obj in _nested_values(sharded_state_dict):
            if isinstance(sh_obj, _ShardedObject):
                sh_obj.replica_id = 0

        if self.do_cache_distribution:
            self.cached_distribution = precomputed_distribution
        end = time.time()
        fp_logger.debug("parallel save sharding, time: %s", end - start)

    return wrapper


def patched_sharded_tensor_to_torch_sharded_tensor_wrapper(original_func):
    """Temporarily skip _parse_and_validate_remote_device during dump.

    Called from mcore_to_pyt_state_dict → _mcore_to_torch_sharded_tensor
    → sharded_tensor_to_torch_sharded_tensor (Megatron torch.py:195).
    Internally calls TorchShardedTensor._init_from_local_shards_and_global_metadata
    which validates placement rank against the process group.

    After WORLD replacement, placement ranks are dump-local (0~3) but the
    process group has global ranks [4,5,6,7] → ValueError.  Replace the
    validation with a no-op within this wrapper's scope.
    """

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        import torch.distributed._shard.sharded_tensor.utils as _stu
        import torch.distributed._shard.sharded_tensor.api as _sta

        _saved_utils = _stu._parse_and_validate_remote_device
        _saved_api = _sta._parse_and_validate_remote_device

        def _noop(pg, remote_device):
            return remote_device.rank(), remote_device.device()

        _stu._parse_and_validate_remote_device = _noop
        _sta._parse_and_validate_remote_device = _noop
        try:
            return original_func(*args, **kwargs)
        finally:
            _stu._parse_and_validate_remote_device = _saved_utils
            _sta._parse_and_validate_remote_device = _saved_api

    return wrapper


def patched_mcore_create_global_plan_wrapper(original_func):
    """Temporarily skip _validate_global_plan during dump.

    Called from save_state_dict_async_plan → MCoreSavePlanner.create_global_plan
    → super().create_global_plan (DefaultSavePlanner) → _create_global_plan
    → _validate_global_plan.  Validates that chunks cover the full tensor
    volume — fails when only dump ranks participate.

    Replace the validation with a no-op within this wrapper's scope.
    """

    @functools.wraps(original_func)
    def wrapper(self, all_plans):
        import torch.distributed.checkpoint.default_planner as _dp

        _saved = _dp._validate_global_plan
        _dp._validate_global_plan = lambda *a, **kw: True
        try:
            return original_func(self, all_plans)
        finally:
            _dp._validate_global_plan = _saved

    return wrapper


def patched_fp_save_wrapper_init_wrapper(original_func):
    """Force FullyParallelSaveStrategyWrapper to use dump_group as parallelization_group.

    verl's save_dist_checkpointing() creates this wrapper with
    mpu.get_data_parallel_group(with_context_parallel=True), which returns
    the original DP-CP group (e.g. [0,4]) containing fault ranks.
    determine_main_replica_uniform_distribution() would hang waiting for
    fault ranks to respond to all_gather_object.

    During dump, replace the group with dump_group so only dump ranks
    participate in the save distribution.
    """

    @functools.wraps(original_func)
    def wrapper(self, strategy, parallelization_group=None, do_cache_distribution=False):
        from mindspeed.core.ttp.replica.replica_group import get_dump_world_group

        _dump_group = get_dump_world_group()
        if _dump_group is not None:
            parallelization_group = _dump_group
        original_func(self, strategy, parallelization_group, do_cache_distribution)

    return wrapper
