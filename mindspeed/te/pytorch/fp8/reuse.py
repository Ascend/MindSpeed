"""Utilities for reusing quantized FP8 weights within one optimizer step."""

from __future__ import annotations

import hashlib
from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed

_WEIGHT_REUSE_POOL: dict[str, Any] = {}
_WEIGHT_REUSE_HITS = 0
_WEIGHT_REUSE_MISSES = 0


def _normalize_hash_value(value: Any) -> Any:
    """Convert kwargs values into a stable, hashable representation."""
    if isinstance(value, dict):
        return tuple((key, _normalize_hash_value(val)) for key, val in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_hash_value(item) for item in value)
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _hash_kwargs(kwargs: dict[str, Any]) -> str:
    normalized_kwargs = tuple(
        (key, _normalize_hash_value(value)) for key, value in sorted(kwargs.items())
    )
    return hashlib.blake2b(
        repr(normalized_kwargs).encode("utf-8"),
        digest_size=8,
    ).hexdigest()


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


def _supports_weight_reuse(tensor: torch.Tensor) -> bool:
    """Only reuse stable weight tensors or views backed by one stable leaf tensor."""
    base_tensor = _get_reuse_base_tensor(tensor)
    return bool(getattr(base_tensor, "is_leaf", False) and getattr(base_tensor, "grad_fn", None) is None)


def generate_weight_reuse_key(
    tensor: torch.Tensor,
    op_name: str,
    kwargs: dict[str, Any],
) -> str:
    """Generate a stable reuse key for a tensor view on the same storage."""
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    base_tensor = _get_reuse_base_tensor(tensor)
    storage = base_tensor.untyped_storage()
    kwargs_hash = _hash_kwargs(kwargs)
    version = getattr(base_tensor, "_version", 0)

    return (
        f"{op_name}:{rank}:{storage.data_ptr()}:{tensor.storage_offset()}:"
        f"{tensor.numel()}:{tuple(tensor.shape)}:{tuple(tensor.stride())}:"
        f"{tensor.device}:{tensor.dtype}:{version}:{kwargs_hash}"
    )


def reuse_or_quantize(
    tensor: torch.Tensor,
    tensor_key: Any,
    quantizer: Callable[..., Any],
    *,
    op_name: str | None = None,
    allow_reuse: bool = True,
    **kwargs: Any,
) -> Any:
    """Run a quantizer and reuse weight results when the feature is enabled."""
    global _WEIGHT_REUSE_HITS, _WEIGHT_REUSE_MISSES

    if (
        not allow_reuse
        or not _is_weight_reuse_enabled(tensor_key)
        or not _supports_weight_reuse(tensor)
    ):
        return quantizer(tensor, **kwargs)

    quantizer_name = op_name or getattr(quantizer, "__name__", quantizer.__class__.__name__)
    cache_key = generate_weight_reuse_key(tensor, quantizer_name, kwargs)
    if cache_key in _WEIGHT_REUSE_POOL:
        _WEIGHT_REUSE_HITS += 1
        return _WEIGHT_REUSE_POOL[cache_key]

    result = quantizer(tensor, **kwargs)
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


def clear_weight_quantization_reuse_cache() -> None:
    """Release cached quantized tensors at the optimizer step boundary."""
    global _WEIGHT_REUSE_HITS, _WEIGHT_REUSE_MISSES

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
    """Clear cached quantized weights around every optimizer step."""

    @wraps(step)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        clear_weight_quantization_reuse_cache()
        try:
            return step(*args, **kwargs)
        finally:
            clear_weight_quantization_reuse_cache()

    return wrapper
