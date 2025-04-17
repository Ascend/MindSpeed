"""Build noop layer of transformer block."""

from typing import Callable, Type

import torch
from mindspeed.core.transformer.transformer_block import _get_layer_offset


def build_with_noop_layers(
    transformer: torch.nn.Module,
    noop_trasformer: Type[torch.nn.Module],
    build_module: Callable,
) -> None:
    """Build transformer layers and handle noop situation.

    Args:
        transformer (torch.nn.Module): A transformer block object which
            contains atrributes:
            - config
            - layers
            - submodules
            - post_process
            - post_layer_norm
            - final_layernorm
        noop_trasformer (Type[torch.nn.Module]): A class to contruct noop
            transformer object.
        build_module (Callable): A function which can
            build module object of torch.

    Returns:
        None: Have no return but modify transformer in place.
    """

    def build_layer(layer_spec, layer_number):
        global_layer_number = _get_layer_offset(transformer.config) + layer_number
        if (
            isinstance(transformer.config.noop_layers, set)
            and global_layer_number - 1 in transformer.config.noop_layers
        ):
            return noop_trasformer(global_layer_number)
        return build_module(
            layer_spec,
            config=transformer.config,
            layer_number=layer_number,
        )

    transformer.layers = torch.nn.ModuleList(
        [
            build_layer(layer_spec, i + 1)
            for i, layer_spec in enumerate(transformer.submodules.layer_specs)
        ]
    )

    if (
        transformer.submodules.layer_norm
        and transformer.post_process
        and transformer.post_layer_norm
    ):
        transformer.final_layernorm = build_module(
            transformer.submodules.layer_norm,
            config=transformer.config,
            hidden_size=transformer.config.hidden_size,
            eps=transformer.config.layernorm_epsilon,
        )
    else:
        transformer.final_layernorm = None  # Either this or nn.Identity
