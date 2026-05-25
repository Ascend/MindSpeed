# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from mindspeed.core.optimizer.muon.muon_utils import (
    get_muon_scale_factor,
    newton_schulz,
    newton_schulz_tp,
)
from mindspeed.core.optimizer.muon.orthogonalized_optimizer import OrthogonalizedOptimizer
from mindspeed.core.optimizer.muon.emerging_optimizers import TensorParallelMuon


__all__ = [
    "OrthogonalizedOptimizer",
    "TensorParallelMuon",
    "get_muon_scale_factor",
    "newton_schulz",
    "newton_schulz_tp",
]
