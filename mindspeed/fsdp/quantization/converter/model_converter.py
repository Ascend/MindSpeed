# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Protocol, Union
import torch.nn as nn

from mindspeed.fsdp.parallel_engine_config import QuantizeConfig


class ModelConverter(Protocol):
    """General model converter interface.
    A model converter is applying a modification to PyTorch model.
    Typical use cases are:
        - Quantization: using QAT, FP8, ... specialized linear layers;
        - Fused optimized layers (e.g. flash-attention, norms, ...)
    """

    def __init__(self, config: QuantizeConfig):
        ...

    def convert(self, model: nn.Module):
        """Inplace conversion of the model."""
        ...

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""
        ...


_registry_model_converter_cls: Dict[str, type[ModelConverter]] = {}
"""Registry of model converter classes.
"""


def register_model_converter(converter_cls: type[ModelConverter], name: str):
    """Register a model converter class.

    A registered model converter can be applied on any model
    using the `model.converters` config parameter.
    """

    if name in _registry_model_converter_cls:
        raise ValueError(f"A model converter '{name}' is already registered.")

    _registry_model_converter_cls[name] = converter_cls


class ModelConvertersContainer(ModelConverter):
    """Model converters sequential container.
    The class build the sequence of model converters defined in `model.converters`
    job config, and apply them to the model sequentially.
    """

    def __init__(self, config: QuantizeConfig):
        converter_classes = [_registry_model_converter_cls[name] for name in config.converters]

        self.converters = [mh_cls(config) for mh_cls in converter_classes]

    def convert(self, model: nn.Module):
        for mh in self.converters:
            mh.convert(model)


def build_model_converter(
        config: QuantizeConfig
) -> ModelConvertersContainer:
    """Build the collection of model converters to apply to the model."""
    return ModelConvertersContainer(config)
