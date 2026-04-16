# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List
import torch.nn as nn

from mindspeed.fsdp.quantization.converter.model_converter import register_model_converter
from mindspeed.fsdp.quantization.converter.utils import module_filter_fn, moe_filter_fn
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class MXLinearConverter:
    """Converts the linear layers of `model` to `MXLinear`."""
    filter_fqns: List[str]
    mx_config: Any  # QuantizeConfig type when imported

    def __init__(self, config: QuantizeConfig):
        # Configure MXFP8
        self.config = config

    def convert(self, model: nn.Module):
        """
        Converts the linear layers of `model` to `MXLinear`.
        Note that today, only MXFP8 (the default) is supported.
        This will mutate the model inplace.
        """

        from mindspeed.fsdp.quantization.converter.utils import convert_model
        from mindspeed.fsdp.quantization.module.linear_mxfp8 import MXLinear
        convert_model(
            model,
            config=self.config,
            convert_fn=MXLinear.from_float,
            filter_fn=module_filter_fn,
            device=model.device,
        )


class MXMoeConverter:
    """Converts the linear layers of `model` to `MXLinear`."""
    filter_fqns: List[str]
    mx_config: Any  # QuantizeConfig type when imported

    def __init__(self, config: QuantizeConfig):
        # Configure MXFP8
        self.config = config

    def convert(self, model: nn.Module):
        """Converts the linear layers of `model` to `MXLinear`.
        Note that today, only MXFP8 (the default) is supported.
        This will mutate the model inplace.
        """
        from mindspeed.fsdp.quantization.converter.utils import convert_model
        from mindspeed.fsdp.quantization.module.gmm_mxfp8 import MXFP8GMM
        convert_model(
            model,
            config=self.config,
            convert_fn=MXFP8GMM.from_float,
            filter_fn=moe_filter_fn,
            device=model.device,
        )


register_model_converter(MXLinearConverter, "quantize.linear.mx")
register_model_converter(MXMoeConverter, "quantize.moe.mx")
