# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, List
import torch.nn as nn

from mindspeed.fsdp.quantization.mxfp8_config import get_mxfp8linear_config
from mindspeed.fsdp.quantization.converter.model_converter import register_model_converter
from mindspeed.fsdp.quantization.utils import module_filter_fn
from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class MXLinearConverter:
    """Converts the linear layers of `model` to `MXLinear`."""
    filter_fqns: List[str]
    mx_config: Any  # MXLinearConfig type when imported

    def __init__(self, config: QuantizeConfig):
        # Configure MXFP8
        self.config = get_mxfp8linear_config(config)

    def convert(self, model: nn.Module):
        """
        Converts the linear layers of `model` to `MXLinear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """

        from mindspeed.fsdp.quantization.quantize import quantize_
        quantize_(
            model,
            config=self.config,
            filter_fn=module_filter_fn,
            device=model.device,
        )


register_model_converter(MXLinearConverter, "quantize.linear.mx")
