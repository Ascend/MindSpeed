# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Utility helpers for accessing model/optimizer/checkpoint from worker context.

Supports both verl Worker instances and TTPReplicaOptimizer instances as
the context object, allowing TTP to work without verl-level patches.
"""

import logging
from typing import Optional

from ..comm.processor import TTPProcessor

logger = logging.getLogger(__name__)


def _find_module_in_chunks(chunks) -> Optional[object]:
    """Extract a module from an iterable of chunks."""
    if not chunks:
        return None
    if not hasattr(chunks, '__iter__'):
        return None
    for chunk in chunks:
        try:
            if hasattr(chunk, '__len__') and len(chunk) > 0:
                return chunk[0]
            elif hasattr(chunk, 'module'):
                return chunk.module
        except Exception:
            logger.warning("[TTP] Failed to iterate chunk", exc_info=True)
    return None


def _get_actor_module_from_worker(worker) -> Optional[object]:
    """Get actor module from worker or optimizer context."""
    if worker is None:
        return _get_module_from_processor()

    # verl worker paths (backward compatible)
    if hasattr(worker, 'actor_module'):
        return worker.actor_module
    try:
        actor = getattr(worker, 'actor', None)
        if actor is not None:
            engine = getattr(actor, 'engine', None)
            if engine is not None:
                return getattr(engine, 'module', None)
    except Exception:
        logger.warning("[TTP] Failed to get actor module from worker", exc_info=True)

    # Megatron optimizer path: extract from model_chunks
    if hasattr(worker, 'model_chunks'):
        try:
            result = _find_module_in_chunks(worker.model_chunks)
            if result is not None:
                return result
        except Exception:
            logger.warning("[TTP] Failed to access model chunks", exc_info=True)

    return _get_module_from_processor()


def _get_actor_optimizer_from_worker(worker) -> Optional[object]:
    """Get actor optimizer from worker or optimizer context."""
    if worker is None:
        return _get_optimizer_from_processor()

    # verl worker paths (backward compatible)
    if hasattr(worker, 'actor_optimizer'):
        return worker.actor_optimizer
    try:
        actor = getattr(worker, 'actor', None)
        if actor is not None:
            engine = getattr(actor, 'engine', None)
            if engine is not None:
                return getattr(engine, 'optimizer', None)
    except Exception:
        logger.warning("[TTP] Failed to get actor optimizer from worker", exc_info=True)

    # Optimizer is the worker itself (TTPReplicaOptimizer path).
    # If this is a sub-optimizer inside a ChainedOptimizer, return the parent
    # (full optimizer) so sharded_state_dict covers all param groups.
    if hasattr(worker, '_ttp_parent_chained'):
        return worker._ttp_parent_chained
    if hasattr(worker, 'ori_dp_group') and hasattr(worker, 'save_parameter_state'):
        return worker

    return _get_optimizer_from_processor()


def _is_actor_param_offload(worker) -> bool:
    """Check if actor params are offloaded to CPU."""
    if worker is None:
        return False

    # verl worker paths (backward compatible)
    if hasattr(worker, '_is_offload_param'):
        return bool(worker._is_offload_param)
    try:
        actor = getattr(worker, 'actor', None)
        if actor is not None:
            engine = getattr(actor, 'engine', None)
            if engine is not None:
                return bool(getattr(engine, '_is_offload_param', False))
    except Exception:
        logger.warning("[TTP] Failed to check actor param offload", exc_info=True)

    # Megatron path: check first parameter's device
    module = _get_actor_module_from_worker(worker)
    if module is not None:
        try:
            return str(next(module.parameters()).device) == 'cpu'
        except Exception:
            logger.warning("[TTP] Failed to check module parameter device", exc_info=True)

    return False


def _get_checkpoint_manager_from_worker(worker) -> Optional[object]:
    """Get checkpoint manager from worker context.

    Supports multiple navigation paths:
    - Direct attribute (mindio-ttp style, worker IS the engine)
    - verl Worker → actor → engine → checkpoint_mananager
    - verl TrainingWorker → engine → checkpoint_mananager
    - TTPProcessor singleton fallback
    """
    if worker is not None:
        # Direct attribute (mindio-ttp style, worker IS the engine)
        if hasattr(worker, 'checkpoint_mananager'):
            return worker.checkpoint_mananager
        if hasattr(worker, 'checkpoint_manager'):
            return worker.checkpoint_manager
        # verl Worker → actor → engine → checkpoint_mananager
        try:
            actor = getattr(worker, 'actor', None)
            if actor is not None:
                engine = getattr(actor, 'engine', None)
                if engine is not None:
                    if hasattr(engine, 'checkpoint_mananager'):
                        return engine.checkpoint_mananager
                    if hasattr(engine, 'checkpoint_manager'):
                        return engine.checkpoint_manager
        except Exception:
            logger.warning("[TTP] Failed to get checkpoint manager from actor", exc_info=True)
        # verl TrainingWorker → engine → checkpoint_mananager
        try:
            engine = getattr(worker, 'engine', None)
            if engine is not None:
                if hasattr(engine, 'checkpoint_mananager'):
                    return engine.checkpoint_mananager
                if hasattr(engine, 'checkpoint_manager'):
                    return engine.checkpoint_manager
        except Exception:
            logger.warning("[TTP] Failed to get checkpoint manager from training worker", exc_info=True)

    # TTPProcessor singleton fallback
    processor = TTPProcessor.get_instance()
    if processor is not None and hasattr(processor, '_checkpoint_manager'):
        return processor._checkpoint_manager
    return None


# ---- Internal fallbacks ----


def _get_module_from_processor():
    """Try to get module from TTPProcessor's optimizer replica info."""
    processor = TTPProcessor.get_instance()
    if processor is not None:
        optimizer = _get_optimizer_from_processor()
        if optimizer is not None and hasattr(optimizer, 'model_chunks'):
            try:
                result = _find_module_in_chunks(optimizer.model_chunks)
                if result is not None:
                    return result
            except Exception:
                logger.warning("[TTP] Failed to access optimizer model chunks", exc_info=True)
    return None


def _get_optimizer_from_processor():
    """Try to get optimizer from TTPProcessor context."""
    processor = TTPProcessor.get_instance()
    if processor is not None:
        worker = processor.save_handler.get_worker_instance()
        if worker is not None:
            return _get_actor_optimizer_from_worker(worker)
    return None
