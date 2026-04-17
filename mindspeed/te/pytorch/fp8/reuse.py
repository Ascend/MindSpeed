# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
"""Utilities for reusing quantized FP8 weights within one optimizer step."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed as dist


_WEIGHT_REUSE_POOL: dict[str, Any] = {}
_RELEASED_BF16_WEIGHT_POOL: dict[int, tuple[torch.Tensor, int, int]] = {}

CacheKey = tuple[Any, ...]
KwargsSignature = tuple[tuple[str, Any], ...]

_EMPTY_KWARGS_SIGNATURE: KwargsSignature = ()
_CACHE_MISS = object()

_WEIGHT_REUSE_HITS = 0
_WEIGHT_REUSE_MISSES = 0
_CACHED_RANK: int | None = None


def _get_rank_fast() -> int:
    global _CACHED_RANK

    rank = _CACHED_RANK
    if rank is not None:
        return rank

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        _CACHED_RANK = rank
        return rank

    return 0


def _normalize_key_value(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            return ()
        return tuple((key, _normalize_key_value(val)) for key, val in sorted(value.items()))
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, tuple):
        return tuple(_normalize_key_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_normalize_key_value(item) for item in value)
    return value


def _make_kwargs_signature(kwargs: dict[str, Any]) -> KwargsSignature:
    if not kwargs:
        return _EMPTY_KWARGS_SIGNATURE
    if len(kwargs) == 1:
        key, value = next(iter(kwargs.items()))
        return ((key, _normalize_key_value(value)),)
    return tuple((key, _normalize_key_value(value)) for key, value in sorted(kwargs.items()))


def _tensor_key_name(tensor_key: Any) -> str:
    return getattr(tensor_key, "value", tensor_key)


def _is_weight_reuse_enabled(tensor_key: Any) -> bool:
    from mindspeed.te.pytorch.fp8.state_manager import FP8GlobalStateManager

    return (
        _tensor_key_name(tensor_key) == "weight"
        and FP8GlobalStateManager.is_weight_quantization_reuse_enabled()
    )


def _get_reuse_base_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._base if getattr(tensor, "_base", None) is not None else tensor


def _is_stable_weight_tensor(tensor: torch.Tensor) -> bool:
    base_tensor = _get_reuse_base_tensor(tensor)
    return bool(getattr(base_tensor, "is_leaf", False) and getattr(base_tensor, "grad_fn", None) is None)



def _get_storage_ptr_for_reuse_key(base_tensor: torch.Tensor) -> int:
    released_weight = _RELEASED_BF16_WEIGHT_POOL.get(id(base_tensor))
    if released_weight is not None:
        _, _, storage_ptr = released_weight
        return storage_ptr
    return base_tensor.untyped_storage().data_ptr()


def _supports_weight_reuse(
    tensor: torch.Tensor,
    reuse_identity: Any = None,
) -> bool:
    if _is_stable_weight_tensor(tensor):
        return True
    return reuse_identity is not None


def generate_weight_reuse_key(
    tensor: torch.Tensor,
    op_name: str,
    reuse_identity: Any = None,
    kwargs: dict[str, Any] = None,
) -> CacheKey:

    if reuse_identity is not None:
        return (
            op_name,
            _get_rank_fast(),
            id(_get_reuse_base_tensor(reuse_identity)) if isinstance(reuse_identity, torch.Tensor) else id(reuse_identity),
            _make_kwargs_signature(kwargs),
        )
    base_tensor = _get_reuse_base_tensor(tensor)

    return (
        op_name,
        _get_rank_fast(),        
        _get_storage_ptr_for_reuse_key(base_tensor),
        tensor.storage_offset(),
        tensor.numel(),
        _make_kwargs_signature(kwargs),

    )


def reuse_or_quantize(
    tensor: torch.Tensor,
    tensor_key: Any,
    quantizer: Callable[..., Any],
    *,
    op_name: str | None = None,
    allow_reuse: bool = True,
    reuse_identity: Any = None,
    **kwargs: Any,
) -> Any:
    global _WEIGHT_REUSE_HITS, _WEIGHT_REUSE_MISSES

    if (
        not allow_reuse
        or not _is_weight_reuse_enabled(tensor_key)
        or not _supports_weight_reuse(tensor, reuse_identity)
    ):
        return quantizer(tensor, **kwargs)

    quantizer_name = op_name or getattr(quantizer, "__name__", quantizer.__class__.__name__)
    cache_key = generate_weight_reuse_key(
        tensor,
        quantizer_name,
        reuse_identity,
        kwargs,
    )

    cached = _WEIGHT_REUSE_POOL.get(cache_key, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        _WEIGHT_REUSE_HITS += 1
        return cached

    result = quantizer(tensor, **kwargs)
    #当前只支持MXFP8场景的释放； 4对应的是双轴量化4个结果：weight_col_mxfp8, weight_col_scale, weight_row_mxfp8, weight_row_scale 
    if isinstance(result, tuple) and len(result) == 4:
        release_bf16_weight_after_quantization(tensor, tensor_key)    
    _WEIGHT_REUSE_POOL[cache_key] = result
    _WEIGHT_REUSE_MISSES += 1
    return result


def _iter_cached_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_cached_tensors(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_cached_tensors(item)


def _supports_bf16_weight_release(tensor: torch.Tensor, tensor_key: Any) -> bool:
    return (
        _is_weight_reuse_enabled(tensor_key)
        and _supports_weight_reuse(tensor)
        and tensor.dtype == torch.bfloat16
    )


def release_bf16_weight_after_quantization(tensor: torch.Tensor, tensor_key: Any) -> None:
    """Release BF16 weight storage after quantization and remember how to restore it."""
    if not _supports_bf16_weight_release(tensor, tensor_key):
        return

    base_tensor = _get_reuse_base_tensor(tensor)
    storage = base_tensor.untyped_storage()
    storage_size = storage.size()
    if storage_size == 0:
        return
    expected_tensor_bytes = base_tensor.numel() * base_tensor.element_size()
    # 防止误杀共享显存(flat_buffer)
    if storage_size > expected_tensor_bytes:
        return

    tensor_id = id(base_tensor)
    if tensor_id not in _RELEASED_BF16_WEIGHT_POOL:
        _RELEASED_BF16_WEIGHT_POOL[tensor_id] = (base_tensor, storage_size, storage.data_ptr())
    storage.resize_(0)


def restore_bf16_weight_storage() -> None:
    """Restore BF16 weight storage before optimizer updates write model weights again."""
    for tensor, storage_size, _ in _RELEASED_BF16_WEIGHT_POOL.values():
        storage = tensor.untyped_storage()
        if storage.size() == storage_size:
            continue
        storage.resize_(storage_size)
    _RELEASED_BF16_WEIGHT_POOL.clear()


def clear_weight_quantization_reuse_cache(release_storage: bool = False) -> None:

    """Release cached quantized tensors at the optimizer step boundary."""
    global _WEIGHT_REUSE_HITS, _WEIGHT_REUSE_MISSES

    if release_storage:
        seen_storage_ptrs: set[int] = set()
        for cached_value in _WEIGHT_REUSE_POOL.values():
            for tensor in _iter_cached_tensors(cached_value):
                storage = tensor.untyped_storage()
                storage_ptr = storage.data_ptr()
                if storage_ptr in seen_storage_ptrs:
                    continue
                seen_storage_ptrs.add(storage_ptr)
                storage.resize_(0)

    _WEIGHT_REUSE_POOL.clear()
    _WEIGHT_REUSE_HITS = 0
    _WEIGHT_REUSE_MISSES = 0


def get_weight_quantization_reuse_stats() -> dict[str, int]:
    return {"hits": _WEIGHT_REUSE_HITS, "misses": _WEIGHT_REUSE_MISSES}


def optimizer_step_reuse_cleanup_wrapper(step: Callable[..., Any]) -> Callable[..., Any]:
    """Clear cached quantized weights before every optimizer step."""

    @wraps(step)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        clear_weight_quantization_reuse_cache()
        restore_bf16_weight_storage()
        return step(*args, **kwargs)


    return wrapper