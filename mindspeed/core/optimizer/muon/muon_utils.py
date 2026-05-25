# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from itertools import chain, cycle, islice, repeat
from typing import Iterator, Literal, Optional, Sequence, Tuple
import warnings

import torch


CoeffIterMode = Literal["cycle", "repeat_last"]
NSCoeffT = Literal["simple", "quintic", "polar_express", "aol", "custom"]
MuonScaleT = Literal["shape_scaling", "spectral", "unit_rms_norm"]


# Coefficient set names and values follow upstream emerging_optimizers.
# polar_express repeats its last coefficient triplet after the listed schedule.
_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


def get_coefficient_iterator(
    steps: int,
    coefficient_sets: Sequence[Tuple[float, float, float]],
    mode: CoeffIterMode = "cycle",
) -> Iterator[Tuple[float, float, float]]:
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if not coefficient_sets:
        raise ValueError("coefficient_sets must be non-empty")

    if mode == "cycle":
        base = cycle(coefficient_sets)
    elif mode == "repeat_last":
        base = chain(coefficient_sets, repeat(coefficient_sets[-1]))
    else:
        raise ValueError(f"Invalid coefficient iteration mode: {mode}")
    return islice(base, steps)


def newton_schulz_step(
    x: torch.Tensor,
    a: float,
    b: float,
    c: float,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Run one PyTorch Newton-Schulz iteration step."""
    A = x @ x.mT
    if tp_group is not None:
        torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    B = b * A + c * (A @ A)
    return a * x + B @ x


def newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: Optional[Sequence[Tuple[float, float, float]]] = None,
    eps: float = 1e-7,
    transpose: Optional[bool] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    use_syrk: bool = False,
) -> torch.Tensor:
    """Compute a Muon-style orthogonalized update with pure PyTorch ops.

    The function returns an FP32 tensor. ``use_syrk`` is accepted for API
    compatibility, but the MindSpeed NPU path falls back to PyTorch matmul.
    """
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    X = x
    if transpose is None:
        transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT

    if tp_group is not None:
        norm = (X * X).sum()
        torch.distributed.all_reduce(norm, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        X = X / torch.sqrt(norm).clamp_min(eps)
    else:
        norm = (X * X).sum(dim=(-2, -1), keepdim=True)
        X = X / torch.sqrt(norm).clamp_min(eps)

    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError("custom_coefficient_sets must be set for coefficient_type='custom'")
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    iter_mode = "repeat_last" if coefficient_type == "polar_express" else "cycle"
    if torch.get_float32_matmul_precision() == "medium":
        if use_syrk:
            warnings.warn(
                "MindSpeed's NPU Newton-Schulz implementation accepts use_syrk "
                "for API compatibility but falls back to PyTorch matmul.",
                UserWarning,
                stacklevel=2,
            )
        X = X.to(torch.bfloat16)
    for a, b, c in get_coefficient_iterator(steps, coefficient_sets, mode=iter_mode):
        X = newton_schulz_step(X, a, b, c, tp_group=tp_group)

    X = X.to(torch.float32)
    if transpose:
        X = X.mT
    return X


def newton_schulz_tp(
    x: torch.Tensor,
    steps: int,
    coefficient_type: NSCoeffT,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: Optional[int] = None,
    tp_mode: Literal["duplicated", "distributed"] = "duplicated",
) -> torch.Tensor:
    """Tensor-parallel Newton-Schulz using only PyTorch distributed collectives."""
    if partition_dim is None:
        return newton_schulz(x, steps, coefficient_type)
    if tp_group is None:
        raise ValueError("tp_group must be set when partition_dim is not None")

    if tp_mode == "duplicated":
        tp_size = tp_group.size()
        tp_rank = tp_group.rank()
        x_shards = [torch.empty_like(x) for _ in range(tp_size)]
        torch.distributed.all_gather(x_shards, x, group=tp_group)
        X = torch.cat(x_shards, dim=partition_dim)
        output = newton_schulz(X, steps, coefficient_type)
        return output.chunk(tp_size, dim=partition_dim)[tp_rank]

    if tp_mode == "distributed":
        if partition_dim == 0:
            transpose = True
        elif partition_dim == 1:
            transpose = False
        else:
            raise ValueError(f"Invalid partition_dim: {partition_dim}")
        return newton_schulz(
            x,
            steps,
            coefficient_type,
            transpose=transpose,
            tp_group=tp_group,
        )

    raise ValueError(f"Invalid tp_mode: {tp_mode}")


def get_muon_scale_factor(size_out: int, size_in: int, mode: MuonScaleT = "spectral") -> float:
    """Return the Muon update scale factor for a matrix shape."""
    if size_out <= 0 or size_in <= 0:
        raise ValueError(f"Muon scale dimensions must be positive, got {size_out}, {size_in}")

    if mode == "shape_scaling":
        return max(1.0, float(size_out) / float(size_in)) ** 0.5
    if mode == "spectral":
        return float(max(size_out, size_in)) ** 0.5
    if mode == "unit_rms_norm":
        return (float(size_out) / float(size_in)) ** 0.5
    raise ValueError(f"Invalid mode for Muon update scale factor: {mode}")


__all__ = [
    "CoeffIterMode",
    "MuonScaleT",
    "NSCoeffT",
    "get_coefficient_iterator",
    "get_muon_scale_factor",
    "newton_schulz",
    "newton_schulz_step",
    "newton_schulz_tp",
]
