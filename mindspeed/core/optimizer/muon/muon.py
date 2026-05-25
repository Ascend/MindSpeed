# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pylint: skip-file

from typing import Any


def get_megatron_muon_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible Muon optimizer getter."""
    from megatron.core.optimizer import get_megatron_optimizer

    use_layer_wise = kwargs.pop("layer_wise_distributed_optimizer", False)

    if "config" in kwargs:
        config = kwargs["config"]
    else:
        config = args[0]

    if use_layer_wise and not getattr(config, "optimizer", "").startswith("dist_"):
        raise ValueError("Layer-wise distributed optimizer is enabled by dist_ prefix in optimizer name.")

    return get_megatron_optimizer(*args, **kwargs)
