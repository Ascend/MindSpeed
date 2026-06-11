# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import gc
import logging
import os
import threading
import time
from typing import Optional

import torch
import torch.distributed as dist

from ..replica.replica_group import (
    set_dump_world_group,
    get_dump_world_group,
    get_dp_cp_ranks,
    ttp_get_replica_dp_num,
)
from ..config import TTPConfig, get_ttp_config
from ..comm.processor import TTPProcessor, register_save_ckpt_handler
from ..utils.worker_utils import (
    _get_actor_module_from_worker,
    _get_actor_optimizer_from_worker,
    _get_checkpoint_manager_from_worker,
    _is_actor_param_offload,
)

logger = logging.getLogger(__name__)

REPLICA_OFFSET = 0


def _reset_replica_groups_for_dump(dump_group_ranks, cur_rank):
    """Replace replica sub-groups so they align with dump_group_ranks."""
    from mindspeed.core.ttp.replica.replica_group import (
        tft_reset_dp_cp_replica_group,
        get_global_dp_cp_ranks,
        get_dp_cp_replica_group,
        get_dp_cp_replica_group_gloo,
        get_replica_num,
    )

    _saved_replica_group = get_dp_cp_replica_group()
    _saved_replica_group_gloo = get_dp_cp_replica_group_gloo()
    _global_dp_ranks = get_global_dp_cp_ranks()
    _replica_num = get_replica_num()
    for _dp_ranks in _global_dp_ranks:
        _replica_group_size = len(_dp_ranks) // _replica_num
        _replica_lists = [
            _dp_ranks[i * _replica_group_size : (i + 1) * _replica_group_size] for i in range(_replica_num)
        ]
        _overlap_lists = [g for g in _replica_lists if set(g) & set(dump_group_ranks)]
        for _rank_list in _overlap_lists:
            _new_group = dist.new_group(_rank_list, use_local_synchronization=True)
            if cur_rank in _rank_list:
                tft_reset_dp_cp_replica_group(_new_group)
    return _saved_replica_group


def _align_optimizer_dp_groups(dump_group_ranks, cur_rank, worker):
    """Reset each sub-optimizer's data_parallel_group to overlap with dump ranks."""
    from mindspeed.core.ttp.replica.replica_group import get_replica_num

    _save_optimizer = _get_actor_optimizer_from_worker(worker)
    if _save_optimizer is None:
        return
    _replica_num = get_replica_num()
    _opt_list = getattr(_save_optimizer, 'chained_optimizers', [_save_optimizer])
    for _sub_opt in _opt_list:
        if not hasattr(_sub_opt, 'ori_dp_list') or not hasattr(_sub_opt, 'data_parallel_group'):
            continue
        _sub_replica_size = len(_sub_opt.ori_dp_list) // _replica_num
        _sub_replica_lists = [
            _sub_opt.ori_dp_list[i : i + _sub_replica_size]
            for i in range(0, len(_sub_opt.ori_dp_list), _sub_replica_size)
        ]
        for _rp_list in _sub_replica_lists:
            _overlap = sorted(set(_rp_list) & set(dump_group_ranks))
            if not _overlap:
                continue
            _new_dp_group = dist.new_group(_overlap, use_local_synchronization=True)
            if cur_rank in _overlap:
                _sub_opt.data_parallel_group = _new_dp_group
                _sub_opt.save_args['rank_list'] = _overlap


def _save_dump_checkpoint(
    step,
    worker,
    actor_local_path,
    dump_group_ranks,
    cur_rank,
    _cached_parallel_state,
    _saved_replica_group,
    global_step,
):
    """Save checkpoint to disk for a fault-triggered dump."""
    _save_optimizer = _get_actor_optimizer_from_worker(worker)

    if _save_optimizer is not None:
        # 4.1 load_megatron_model_to_gpu
        _actor_module = _get_actor_module_from_worker(worker)
        _need_load = _is_actor_param_offload(worker)
        if not _need_load and _actor_module is not None:
            try:
                _device = str(next(_actor_module.parameters()).device)
                _need_load = _device == "cpu"
            except Exception as _device_check_e:
                logger.debug("[TTP] Failed to check model device: %s", _device_check_e)
        if _need_load and _actor_module is not None:
            try:
                _device = torch.cuda.current_device() if torch.cuda.is_available() else 'npu'
                _actor_module.to(_device)
            except Exception as _load_gpu_e:
                logger.warning("[TTP] Failed to load model to GPU: %s", _load_gpu_e)

        # 4.2 NPU stop_device + restart_device
        try:
            _pause_wait = 3.0
            time.sleep(_pause_wait)
            gc.collect()
        except Exception:
            logger.warning("[TTP] GC collect failed during dump save", exc_info=True)

        try:
            import torch_npu

            _device = torch.npu.current_device()
            torch.npu.set_device(_device)
            torch_npu.npu.stop_device(_device)
            torch.npu.restart_device(_device)
            if hasattr(torch.npu, 'SyncLaunchStream'):
                torch.npu.SyncLaunchStream()
            else:
                torch.npu.synchronize()
        except ImportError:
            pass
        except Exception:
            logger.warning("[TTP] Device stop/restart failed during dump save", exc_info=True)

        # 4.4-4.9: Apply Megatron-layer dump patches
        from mindspeed.features_manager.ttp.dump_patches import apply_dump_patches

        apply_dump_patches()

        # 4.4.5 Set dump args on each sub-optimizer
        _opt_list = getattr(_save_optimizer, 'chained_optimizers', [_save_optimizer])
        for _sub_opt in _opt_list:
            if hasattr(_sub_opt, 'set_dump_args'):
                try:
                    _sub_opt.set_dump_args(
                        rank=_sub_opt.save_args.get('rank', cur_rank),
                        step=step,
                        rank_list=dump_group_ranks,
                        global_rank=cur_rank,
                    )
                except Exception:
                    logger.warning("[TTP] Failed to apply dump patch", exc_info=True)

        # 4.5 Find checkpoint manager
        _ckpt_mgr = _get_checkpoint_manager_from_worker(worker)
        if _ckpt_mgr is None:
            _ckpt_mgr = _find_checkpoint_manager_via_gc(_save_optimizer)
            if _ckpt_mgr is None:
                logger.error(
                    "[SAVE_CALLBACK_ERROR] Cannot get checkpoint_mananager via gc fallback: worker=%s",
                    type(worker).__name__ if worker else None,
                )
                _restore_parallel_state_cache(_cached_parallel_state)
                return False

        # Override rank for dump
        _saved_rank = _ckpt_mgr.rank
        if cur_rank == dump_group_ranks[0]:
            _ckpt_mgr.rank = 0
        _ckpt_mgr.checkpoint_config.async_save = False
        _saved_save_contents = _ckpt_mgr.checkpoint_save_contents[:]
        _ckpt_mgr.checkpoint_save_contents = ["model", "optimizer", "extra"]

        try:
            _ckpt_mgr.save_checkpoint(
                local_path=actor_local_path,
                hdfs_path=None,
                global_step=global_step,
                max_ckpt_to_keep=None,
            )
        finally:
            _ckpt_mgr.rank = _saved_rank
            _ckpt_mgr.checkpoint_save_contents = _saved_save_contents

        # 6.2.1 Restore REPLICA sub-group
        try:
            from mindspeed.core.ttp.replica.replica_group import tft_reset_dp_cp_replica_group

            tft_reset_dp_cp_replica_group(_saved_replica_group)
        except Exception:
            logger.warning("[TTP] Failed to restore replica group", exc_info=True)

        # 6.3 Restore parallel_state cache
        _restore_parallel_state_cache(_cached_parallel_state)

        # offload model to CPU
        if _need_load and _actor_module is not None:
            try:
                _actor_module.to('cpu')
            except Exception as _offload_e:
                logger.debug("[TTP] model offload to CPU failed: %s", _offload_e)
    else:
        logger.error("[SAVE_CALLBACK_ERROR] Cannot get optimizer for dump, aborting")
        _restore_parallel_state_cache(_cached_parallel_state)
        return False

    return True


def _find_checkpoint_manager_via_gc(_save_optimizer):
    """Search all live objects for a MegatronCheckpointManager instance."""
    _candidates = []
    for _obj in gc.get_objects():
        if _obj.__class__.__name__ == 'MegatronCheckpointManager':
            _candidates.append(_obj)
    if len(_candidates) == 1:
        return _candidates[0]
    if len(_candidates) > 1:
        for _mgr in _candidates:
            try:
                if hasattr(_mgr, 'role') and _mgr.role == 'actor' and _mgr.optimizer is _save_optimizer:
                    return _mgr
            except Exception:
                logger.warning("[TTP] Failed to inspect gc object for checkpoint manager", exc_info=True)
        return _candidates[0]
    return None


def _cache_parallel_state():
    """Cache parallel_state rank/world_size values during dump.

    Prevents errors from dist.get_rank(group=tp_group) when default_pg is replaced.
    These values are invariant and depend only on model parallel topology, not dump_group.
    Must be called before _replace_default_group_with_dump.
    """
    import megatron.core.parallel_state as ps

    cached = {}

    if ps._MPU_TENSOR_MODEL_PARALLEL_RANK is None:
        cached['tp_rank'] = ps.get_tensor_model_parallel_rank()
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = cached['tp_rank']
    if ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
        cached['tp_size'] = ps.get_tensor_model_parallel_world_size()
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = cached['tp_size']
    if ps._MPU_PIPELINE_MODEL_PARALLEL_RANK is None:
        cached['pp_rank'] = ps.get_pipeline_model_parallel_rank()
        ps._MPU_PIPELINE_MODEL_PARALLEL_RANK = cached['pp_rank']
    if ps._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is None:
        cached['pp_size'] = ps.get_pipeline_model_parallel_world_size()
        ps._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = cached['pp_size']
    if ps._MPU_DATA_PARALLEL_RANK is None:
        cached['dp_rank'] = ps.get_data_parallel_rank(with_context_parallel=True)
        ps._MPU_DATA_PARALLEL_RANK = cached['dp_rank']
    if ps._MPU_DATA_PARALLEL_WORLD_SIZE is None:
        cached['dp_size'] = ps.get_data_parallel_world_size(with_context_parallel=True)
        ps._MPU_DATA_PARALLEL_WORLD_SIZE = cached['dp_size']

    return cached


def _restore_parallel_state_cache(cached):
    """Restore cached parallel_state variables to None. Only restores values set by this patch."""
    if not cached:
        return
    import megatron.core.parallel_state as ps

    if 'tp_rank' in cached:
        ps._MPU_TENSOR_MODEL_PARALLEL_RANK = None
    if 'tp_size' in cached:
        ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    if 'pp_rank' in cached:
        ps._MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    if 'pp_size' in cached:
        ps._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    if 'dp_rank' in cached:
        ps._MPU_DATA_PARALLEL_RANK = None
    if 'dp_size' in cached:
        ps._MPU_DATA_PARALLEL_WORLD_SIZE = None


def _replace_default_group_with_dump(dump_group):
    # Replace PyTorch default process group with dump_group so ungrouped collectives use it
    original_world = dist.GroupMember.WORLD
    dist.GroupMember.WORLD = dump_group
    return original_world


def _restore_default_group(original_world):
    dist.GroupMember.WORLD = original_world


def tft_init_controller_processor(config: Optional[TTPConfig] = None):
    if config is None:
        config = get_ttp_config()

    if not config.enabled:
        logger.warning("[TTP] TTP is disabled, skipping initialization")
        return

    if not dist.is_initialized():
        logger.warning("[TTP] torch.distributed not initialized, TTP will use default rank 0")
        rank = 0
        world_size = 1
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    os.environ["TTP_STOP_CLEAN_BEFORE_DUMP"] = "1"

    server_ip = os.environ.get('TTP_SERVER_IP', config.server_ip)
    server_port = int(os.environ.get('TTP_SERVER_PORT', config.server_port))

    # rank 0: start Controller thread before Processor
    if rank == 0:
        logger.warning("[TTP] rank=0 starting Controller thread on %s:%d", server_ip, server_port)
        from mindspeed.core.ttp.comm.controller import TTPController

        controller = TTPController(rank=0, world_size=world_size, config=config)
        t = threading.Thread(target=controller.start, args=(server_ip, server_port), daemon=True)
        t.start()
        if not controller.wait_ready(timeout=10.0):
            logger.error("[TTP] Controller failed to start within 10s")
            raise RuntimeError("Controller failed to start within 10 seconds")
        logger.warning("[TTP] Controller thread ready on %s:%d", server_ip, server_port)

    existing_processor = TTPProcessor.get_instance()
    if existing_processor is not None:
        logger.warning("[TTP] TTPProcessor already initialized for rank=%s, skipping", rank)
    else:
        processor = TTPProcessor(rank, world_size, config)
        processor.start(server_ip, server_port)
        TTPProcessor.set_instance(processor)
        logger.warning("[TTP] TTP Processor started on rank %s", rank)


def tft_register_processor():
    tft_set_optimizer_replica()
    register_save_ckpt_handler(tft_save_callback)


def tft_set_optimizer_replica():
    replica_info = []
    dp_cp_ranks = get_dp_cp_ranks()
    dense_replica_cnt = ttp_get_replica_dp_num()
    replica_offset = REPLICA_OFFSET

    if dp_cp_ranks:
        replica_dict = {"rank_list": dp_cp_ranks, "replica_cnt": dense_replica_cnt, "replica_shift": replica_offset}
        replica_info.append(replica_dict)

    if replica_info:
        processor = TTPProcessor.get_instance()
        if processor:
            processor.save_handler.set_optimizer_replica(replica_info)


def tft_save_callback(step: int, save_info: list, worker=None):
    """Save a checkpoint during a fault (dump) using the replica group topology.

    This is the main dump save entry point. It:
    1. Creates a dump_group from the world_ranks in save_info.
    2. Replaces the default PyTorch process group and replica groups for the save.
    3. Patches various Megatron/torch internals to bypass validation checks.
    4. Saves the checkpoint via checkpoint_manager.save_checkpoint().
    5. Restores all patches and groups after the save.

    Args:
        step: Current training step.
        save_info: List of dicts with dump rank information from the Controller.
        worker: The verl Worker instance holding the actor module and checkpoint manager.

    Returns:
        True if the checkpoint was saved successfully, False otherwise.
    """
    cur_rank = dist.get_rank() if dist.is_initialized() else 0

    # ===== 1. Get dump_group_ranks =====
    dump_group_ranks = save_info[0].get("world_ranks", None) if save_info else None

    if dump_group_ranks is None:
        try:
            from mindspeed.core.ttp.comm.processor import get_processor

            processor = get_processor()
            if processor and hasattr(processor, 'save_handler'):
                dump_group_ranks = processor.save_handler.get_dump_world_ranks()
                if dump_group_ranks:
                    pass
        except Exception as e:
            logger.debug("[TTP] get dump_group_ranks from save_handler failed: %s", e)

    if dump_group_ranks is None:
        logger.error("[SAVE_CALLBACK_ERROR] Cannot get dump_group_ranks from save_info or save_handler")
        return False

    # ===== 2. Create dump_group =====
    dump_group = dist.new_group(dump_group_ranks, use_local_synchronization=True)
    set_dump_world_group(dump_group)

    # 2.1 Cache parallel_state rank/world_size (must occur before replacing default_pg)
    _cached_parallel_state = _cache_parallel_state()

    # 2.2 Replace PyTorch default group so ungrouped collectives use dump_group
    original_world = _replace_default_group_with_dump(dump_group)
    try:
        # Replace patch_mpu_for_dump with tft_reset_dp_cp_replica_group.
        _saved_replica_group = _reset_replica_groups_for_dump(dump_group_ranks, cur_rank)

        # ===== 2.4 Align sub-optimizer data_parallel_group with dump ranks =====
        _align_optimizer_dp_groups(dump_group_ranks, cur_rank, worker)

        # ===== 3. Compute global_step and path =====
        if worker is not None and hasattr(worker, 'config') and hasattr(worker.config, 'actor'):
            ppo_epochs = worker.config.actor.ppo_epochs
            global_step = step // ppo_epochs
        else:
            global_step = step

        config = get_ttp_config()
        dump_dir = config.dump_dir
        local_global_step_folder = os.path.join(dump_dir, f"global_step_{global_step}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        # ===== 4. Save checkpoint =====
        if not _save_dump_checkpoint(
            step,
            worker,
            actor_local_path,
            dump_group_ranks,
            cur_rank,
            _cached_parallel_state,
            _saved_replica_group,
            global_step,
        ):
            return False

        # ===== 7. Barrier and cleanup =====
        try:
            _dump_group_for_barrier = get_dump_world_group()
            if _dump_group_for_barrier is not None:
                dist.barrier(group=_dump_group_for_barrier)
        except Exception:
            logger.warning("[TTP] Barrier failed during dump save cleanup", exc_info=True)

        set_dump_world_group(None)

        # ===== 8. Verify dump success =====
        dump_path = os.path.join(dump_dir, f"global_step_{global_step}")
        actor_path = os.path.join(dump_path, "actor")
        dist_ckpt_path = os.path.join(actor_path, "dist_ckpt")  # verl creates this subdir

        # PyTorch DCP writes .metadata inside the save directory, not
        # actor_path/metadata.json (which is Megatron content metadata).
        metadata_path = os.path.join(dist_ckpt_path, ".metadata")
        metadata_exists = os.path.exists(metadata_path)

        common_path = os.path.join(dist_ckpt_path, "common.pt")
        common_exists = os.path.exists(common_path)

        # Also verify at least one distcp file exists
        distcp_files = (
            [f for f in os.listdir(dist_ckpt_path) if f.endswith('.distcp')] if os.path.isdir(dist_ckpt_path) else []
        )
        distcp_count = len(distcp_files)

        dump_success = metadata_exists and common_exists and distcp_count > 0

        return dump_success

    finally:
        _restore_default_group(original_world)
