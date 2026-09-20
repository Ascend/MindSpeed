# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""Block-level CPU offload for arbitrary tensors (``block_swap``).

Public entry points:
    - ``block_swap``: context manager - materialize device tensors inside a
      ``with`` scope, release the storage on exit (training-safe via
      saved-tensor hooks).
    - ``block_swap_all_in`` / ``block_swap_all_out``: manual bulk control.
    - ``SwapState`` / ``get_swap_manager``: per-tensor swap primitive and the
      process-wide registry.
"""

from mindspeed.core.memory.block_swap.block_swap import (
    block_swap,
    block_swap_all_in,
    block_swap_all_out,
)
from mindspeed.core.memory.block_swap.swap_state import (
    SwapState,
    get_swap_manager,
)

__all__ = [
    'block_swap',
    'block_swap_all_in',
    'block_swap_all_out',
    'SwapState',
    'get_swap_manager',
]
