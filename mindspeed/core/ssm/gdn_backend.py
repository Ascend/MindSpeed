# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
