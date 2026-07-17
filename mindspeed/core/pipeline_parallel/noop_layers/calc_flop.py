"""Calculate total number of floating point openrations."""

from argparse import Namespace
from typing import Callable


def calc_flop(func: Callable, args: Namespace, batch_size: int, *flop_args, **flop_kwargs) -> float:
    """Calculate total number of floating point operations of the model
        by consider the noop transformer layers situation.

    Args:
        func (Callable): A function to calculate flop
            without considering noop tranformer layer.
            the func have two arguments such as `func(args, batch_size)`.
        args (Namespace): Arguments from cli or configure file.
        batch_size (int): batch size of dataset.

    Returns:
        float: Total number of floating point operations
            considering noop transformer layers.
    """
    is_noop_layers_set = isinstance(args.noop_layers, set)
    noop_layer_count = len(args.noop_layers) if is_noop_layers_set else 0
    args.num_layers -= noop_layer_count
    try:
        return func(args, batch_size, *flop_args, **flop_kwargs)
    finally:
        args.num_layers += noop_layer_count
