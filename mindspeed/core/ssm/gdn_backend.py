# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Direct and atomic backend selection for Gated DeltaNet operators."""

from dataclasses import dataclass
import logging
from typing import Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GdnOperators:
    name: str
    causal_conv1d: Callable
    l2norm: Callable
    chunk_gated_delta_rule: Callable


def _load_fla_npu_operators():
    from fla_npu.ops.triton import l2norm
    from mindspeed.core.ssm.npu_causal_conv1d import causal_conv1d
    from mindspeed.core.ssm.npu_chunk_gated_delta_rule import chunk_gated_delta_rule

    return GdnOperators("fla_npu", causal_conv1d, l2norm, chunk_gated_delta_rule)


def _load_mindspeed_operators():
    from mindspeed.core.ssm.torch_causal_conv1d import causal_conv1d
    from mindspeed.core.ssm.triton_chunk_gated_delta_rule import chunk_gated_delta_rule
    from mindspeed.ops.triton.l2norm import l2norm

    return GdnOperators("mindspeed", causal_conv1d, l2norm, chunk_gated_delta_rule)


def load_gdn_operators() -> GdnOperators:
    try:
        operators = _load_fla_npu_operators()
    except ImportError as error:
        logger.warning(
            "FLA-NPU GDN backend import failed; using MindSpeed backend: %s",
            error,
        )
        operators = _load_mindspeed_operators()
    logger.info("MindSpeed GDN selected backend: %s", operators.name)
    return operators
