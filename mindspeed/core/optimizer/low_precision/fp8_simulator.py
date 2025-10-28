"""
Utilities to exercise the FP8 optimiser path without requiring NPU hardware.

When ``torch_npu`` is unavailable �?or when the environment variable
``MINDSPEED_FP8_SIM`` is set to a truthy value �?this module provides
lightweight stand-ins for the FP8 quantisation entry points that would normally
be served by Huawei's NPU runtime.  It also detects at runtime when the NPU
driver reports that float8 is unsupported and automatically falls back to the
software implementation so that training can continue (albeit without the memory
benefits of true FP8 storage).

Typical usage::

    from mindspeed.core.optimizer.low_precision.fp8_simulator import install_simulated_torch_npu

    install_simulated_torch_npu(force=True)
    # imports that expect torch_npu can now be safely evaluated

Alternatively, rely on :func:`get_torch_npu` inside the low-precision optimiser
stack, or export ``MINDSPEED_FP8_SIM=1`` before launching a test run.
"""

from __future__ import annotations

import math
import os
import sys
import types
import threading
from typing import Callable, Optional, Tuple

import torch

__all__ = [
    "get_torch_npu",
    "install_simulated_torch_npu",
    "cast_with_fp8_guard",
    "set_hifloat8_emulation_max",
]

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    return default

_FLOAT8_DTYPES = {
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
}
_CACHED_PROXY = None
_FLOAT8_SUPPORT: dict[torch.dtype, bool] = {}
_THREAD_CONTEXT = threading.local()
_MXFP8_DEBUG = _env_flag("MINDSPEED_MXFP8_DEBUG")
_LINEAR_FP8_SCALE = _env_flag("MINDSPEED_FP8_SIM_LINEAR_SCALE", True)
_POW2_SCALE_MULT = float(os.environ.get("MINDSPEED_FP8_SIM_SCALE_MULT", "128.0"))


def _should_force_simulation() -> bool:
    return _env_flag("MINDSPEED_FP8_SIM")


def _resolve_dst_type(dst_type: Optional[torch.dtype], force_fp32: bool = False) -> torch.dtype:
    """Map the requested dtype to one that is always safe."""
    if dst_type is None or force_fp32:
        return torch.float32
    if dst_type in _FLOAT8_DTYPES:
        return torch.float32
    try:
        torch.empty(1, dtype=dst_type)
        return dst_type
    except (TypeError, RuntimeError):
        return torch.float32


def _safe_absmax(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.ones(1, dtype=torch.float32, device=tensor.device)
    amax = tensor.abs().max()
    if torch.isfinite(amax) and amax > 0:
        return amax
    return torch.ones(1, dtype=torch.float32, device=tensor.device)


def _simulate_fp8_with_cpu(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if target_dtype not in _FLOAT8_DTYPES:
        return _safe_to_dtype(tensor, out_dtype)
    cpu_tensor = tensor.detach().to("cpu")
    try:
        fp8_tensor = cpu_tensor.to(target_dtype)
    except RuntimeError as err:
        if _error_indicates_missing_fp8(err):
            return _safe_to_dtype(cpu_tensor, out_dtype, device=tensor.device)
        raise
    simulated = fp8_tensor.to(out_dtype)
    return simulated.to(tensor.device)


def _quantize_blockwise(
    tensor: torch.Tensor,
    block_size: Optional[int],
    dst_type: Optional[torch.dtype],
    force_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tensor_fp32 = tensor.to(torch.float32)
    dst_type_resolved = _resolve_dst_type(dst_type, force_fp32)

    if block_size is None or block_size < 1:
        scale_value = _safe_absmax(tensor_fp32)
        quant = (tensor_fp32 / scale_value).to(dst_type_resolved)
        scale = scale_value.reshape(1).to(torch.float32)
        return quant, scale
    if tensor_fp32.numel() == 0:
        empty = torch.empty_like(tensor_fp32, dtype=dst_type_resolved)
        return empty, torch.ones(1, dtype=torch.float32, device=tensor.device)

    axis_size = tensor_fp32.shape[-1] if tensor_fp32.dim() > 0 else tensor_fp32.numel()
    leading = tensor_fp32.reshape(-1, axis_size)
    rows, axis_dim = leading.shape

    blocks_per_row = 0 if block_size == 0 else axis_dim // block_size
    tail = axis_dim % block_size
    num_blocks = rows * blocks_per_row + (rows if tail > 0 else 0)
    if num_blocks == 0:
        scales = torch.ones(1, dtype=torch.float32, device=tensor.device)
        return tensor_fp32.to(dst_type_resolved), scales

    scales = torch.empty(num_blocks, dtype=torch.float32, device=tensor.device)
    quant_rows = torch.empty_like(leading, dtype=dst_type_resolved)
    offset = 0

    def _compute_scale(values: torch.Tensor) -> torch.Tensor:
        safe = torch.where(
            (values > 0) & torch.isfinite(values),
            values,
            torch.ones_like(values, dtype=torch.float32),
        ).to(torch.float32)
        if dst_type in _FLOAT8_DTYPES and not _LINEAR_FP8_SCALE:
            log2 = torch.log2(safe)
            log2 = torch.ceil(log2)
            log2 = torch.clamp(log2, min=-126.0, max=128.0)
            scale = torch.pow(2.0, log2) * _POW2_SCALE_MULT
            return scale.to(torch.float32)
        return safe

    if blocks_per_row > 0:
        view_len = blocks_per_row * block_size
        blocks = leading[:, :view_len].reshape(rows, blocks_per_row, block_size)
        block_amax = blocks.abs().amax(dim=2)
        block_scale = _compute_scale(block_amax)
        scales[: rows * blocks_per_row] = block_scale.reshape(-1)
        quant_blocks = (blocks / block_scale.unsqueeze(-1)).to(dst_type_resolved)
        quant_rows[:, :view_len] = quant_blocks.reshape(rows, view_len)
        offset = rows * blocks_per_row

    if tail > 0:
        tail_view = leading[:, -tail:]
        tail_amax = tail_view.abs().amax(dim=1)
        tail_scale = _compute_scale(tail_amax)
        scales[offset:] = tail_scale
        quant_tail = (tail_view / tail_scale.unsqueeze(-1)).to(dst_type_resolved)
        quant_rows[:, -tail:] = quant_tail

    quant = quant_rows.reshape_as(tensor_fp32)
    return quant, scales


def _encode_mxfp8_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.numel() == 0:
        return torch.ones(1, dtype=torch.float32, device=scale.device)

    scale_fp32 = scale.to(torch.float32)
    safe_scale = torch.where(
        torch.isfinite(scale_fp32) & (scale_fp32 > 0),
        scale_fp32,
        torch.ones_like(scale_fp32),
    )
    safe_scale_inv = torch.where(
        safe_scale > 0,
        1.0 / safe_scale,
        torch.ones_like(safe_scale),
    )
    codes = torch.log2(safe_scale_inv) + 127.0
    codes = torch.clamp(torch.round(codes), min=0.0, max=254.0)
    if _MXFP8_DEBUG:
        sample_count = min(8, safe_scale.numel())
        sample_scale = safe_scale.view(-1)[:sample_count].cpu()
        sample_codes = codes.view(-1)[:sample_count].cpu()
        print(
            "[MXFP8-DEBUG simulator] scale head:",
            sample_scale.tolist(),
            "codes head:",
            sample_codes.tolist(),
            flush=True,
        )
    return codes.to(dtype=torch.float32)


def _npu_dynamic_mx_quant(
    tensor: torch.Tensor,
    block_size: Optional[int] = None,
    dst_type: Optional[torch.dtype] = None,
):
    quant, scale = _quantize_blockwise(tensor, block_size, dst_type, force_fp32=True)
    if scale.numel() == 0:
        scale = torch.ones(1, dtype=torch.float32, device=tensor.device)
    if dst_type in _FLOAT8_DTYPES:
        quant_tensor = _simulate_fp8_with_cpu(quant, dst_type, torch.float16)
        if _LINEAR_FP8_SCALE:
            linear_scale = torch.where(
                torch.isfinite(scale) & (scale > 0),
                scale,
                torch.ones_like(scale, dtype=torch.float32),
            ).to(torch.float32)
            return quant_tensor, linear_scale
        encoded_scale = _encode_mxfp8_scale(scale)
        return quant_tensor, encoded_scale
    return quant.to(_resolve_dst_type(dst_type, force_fp32=True)), scale


class _HiFloat8Tensor:
    @staticmethod
    def to_hifloat8(tensor: torch.Tensor) -> torch.Tensor:
        tensor_fp32 = tensor.to(torch.float32)
        hi_max = float(max(1.0, getattr(_THREAD_CONTEXT, "hif8_max", 15.0)))
        hi_min = -hi_max
        clamped = torch.clamp(tensor_fp32, min=hi_min, max=hi_max)
        # Simulate symmetric 8-bit quantisation that matches the hardware range
        # exposed by HiFloat8.  The real kernel maps values linearly into an
        # 8-bit code space, so we emulate that by rounding to the nearest code
        # and rescales back to FP16.
        max_code = 127.0
        scale = hi_max / max_code
        inv_scale = 0.0 if scale == 0.0 else 1.0 / scale
        quant_int = torch.round(clamped * inv_scale)
        quant_int = torch.clamp(quant_int, min=-max_code, max=max_code)
        dequant = quant_int * scale
        return dequant.to(torch.float16)


def set_hifloat8_emulation_max(max_value: Optional[float]) -> None:
    """
    Hint the simulator about the target HiFloat8 dynamic range.
    Real torch_npu kernels receive the expected format implicitly. When the
    simulator is active we rely on this setter so that the emulation can choose
    between the hif8_15 and hif8_224 ranges.
    """
    if max_value is None:
        if hasattr(_THREAD_CONTEXT, "hif8_max"):
            delattr(_THREAD_CONTEXT, "hif8_max")
        return
    _THREAD_CONTEXT.hif8_max = float(max_value)


def _error_indicates_missing_fp8(err: RuntimeError) -> bool:
    message = str(err).lower()
    if "fill_empty_deterministic_" in message:
        return True
    if "not implemented" in message and "float8" in message:
        return True
    return "not been supported" in message or "not support" in message or "unsupported" in message


def _deterministic_mode_enabled() -> bool:
    checker = getattr(torch, "are_deterministic_algorithms_enabled", None)
    if callable(checker):
        try:
            if checker():
                return True
        except TypeError:
            pass
    for key in ("TORCH_NPU_DETERMINISTIC", "NPU_DETERMINISTIC"):
        if os.environ.get(key, "").strip().lower() in _TRUE_STRINGS:
            return True
    return False


def _safe_to_dtype(tensor: torch.Tensor, dst_dtype: torch.dtype, device: Optional[torch.device] = None) -> torch.Tensor:
    target_device = device or tensor.device
    if tensor.dtype == dst_dtype:
        return tensor.to(device=target_device)
    if not torch.is_floating_point(tensor):
        return tensor.to(device=target_device, dtype=dst_dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dst_dtype)):
        return tensor.to(device=target_device, dtype=dst_dtype)
    try:
        finfo = torch.finfo(dst_dtype)
    except (TypeError, ValueError):
        return tensor.to(device=target_device, dtype=dst_dtype)

    abs_max_tensor = tensor.detach().abs().max()
    abs_max_val = abs_max_tensor.item() if abs_max_tensor.numel() == 1 else float("inf")
    if math.isfinite(abs_max_val) and abs_max_val <= finfo.max:
        return tensor.to(device=target_device, dtype=dst_dtype)
    return tensor.to(device=target_device, dtype=torch.float32)


def _build_hi_float_proxy(real_cls, force_sim: bool):
    class HiFloat8TensorProxy:
        @staticmethod
        def to_hifloat8(tensor: torch.Tensor) -> torch.Tensor:
            if not force_sim and real_cls is not None and hasattr(real_cls, "to_hifloat8"):
                try:
                    return real_cls.to_hifloat8(tensor)
                except RuntimeError as err:
                    if not _error_indicates_missing_fp8(err):
                        raise
            quant, _ = _quantize_blockwise(tensor, None, torch.float32, force_fp32=True)
            target_dtype = getattr(torch, "float8_e4m3fn", None)
            if target_dtype is None:
                return quant.to(torch.float16)
            return _simulate_fp8_with_cpu(quant, target_dtype, torch.float16)

    return HiFloat8TensorProxy


def _build_quant_proxy(real_func: Optional[Callable], force_sim: bool):
    def _proxy(tensor: torch.Tensor, block_size: Optional[int] = None, dst_type: Optional[torch.dtype] = None):
        if not force_sim and callable(real_func):
            try:
                return real_func(tensor, block_size=block_size, dst_type=dst_type)
            except RuntimeError as err:
                if not _error_indicates_missing_fp8(err):
                    raise
        return _npu_dynamic_mx_quant(tensor, block_size=block_size, dst_type=dst_type)

    return _proxy


def _build_module(real_module=None, force_sim: bool = False):
    module = types.ModuleType("torch_npu_sim")
    if real_module is not None:
        module.__dict__.update(getattr(real_module, "__dict__", {}))

    module.HiFloat8Tensor = _build_hi_float_proxy(
        getattr(real_module, "HiFloat8Tensor", None),
        force_sim,
    )
    module.npu_dynamic_mx_quant = _build_quant_proxy(
        getattr(real_module, "npu_dynamic_mx_quant", None),
        force_sim,
    )
    module.__dict__.setdefault("__wrapped__", real_module)
    return module


def get_torch_npu():
    """Return a torch_npu-like module with FP8 fallbacks when necessary."""
    global _CACHED_PROXY
    if _CACHED_PROXY is not None:
        return _CACHED_PROXY

    real_module = None
    try:
        import torch_npu as _torch_npu  # type: ignore

        real_module = _torch_npu
    except ImportError:
        real_module = None

    proxy = _build_module(real_module, force_sim=_should_force_simulation())
    _CACHED_PROXY = proxy
    return proxy


def install_simulated_torch_npu(force: bool = False):
    """
    Install a simulated ``torch_npu`` module into ``sys.modules``.

    This can be used to run import-time checks that expect ``torch_npu`` to be
    present even when the real package is missing.
    """
    global _CACHED_PROXY
    real_module = sys.modules.get("torch_npu")
    force = force or _should_force_simulation()
    proxy = _build_module(real_module, force_sim=force)
    sys.modules["torch_npu"] = proxy
    _CACHED_PROXY = proxy
    return proxy


def cast_with_fp8_guard(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    fallback_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Attempt to cast ``tensor`` to ``target_dtype`` and automatically fall back to
    ``fallback_dtype`` (default: float32) if the runtime reports that float8 is
    unsupported.
    """
    if target_dtype not in _FLOAT8_DTYPES:
        return tensor.to(target_dtype)

    support = _FLOAT8_SUPPORT.get(target_dtype)
    if support is False:
        return _simulate_fp8_with_cpu(tensor, target_dtype, fallback_dtype)

    try:
        result = tensor.to(target_dtype)
        _FLOAT8_SUPPORT[target_dtype] = True
        return result
    except RuntimeError as err:
        if not _error_indicates_missing_fp8(err):
            raise
        _FLOAT8_SUPPORT[target_dtype] = False
        return _simulate_fp8_with_cpu(tensor, target_dtype, fallback_dtype)




