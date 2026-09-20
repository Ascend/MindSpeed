# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""Per-tensor swap primitive: pinned host mirror + on-demand H2D/D2H +
device-storage release (``storage().resize_(0)``), following the proven
swap-optimizer pattern (mindspeed/core/optimizer/swap_optimizer).

Version-counter safety: materialization refills through ``param.data.copy_()``
which does not bump ``param._version``, so autograd's in-place checks on
``ctx.saved_tensors`` keep passing.
"""

import torch


class SwapState:
    """Swap state attached to one managed tensor.

    ``swap_in``/``swap_out`` are raw primitives (used by the optimizer-update
    hooks); ``acquire``/``release`` are depth-counted scopes (used by
    ``block_swap``, so nested blocks release only at the outermost exit).
    """

    def __init__(self, param):
        self.param = param
        # pinned host mirror (same shape/dtype), filled once here
        self.cpu = torch.empty_like(param, pin_memory=True, device='cpu')
        self.cpu.copy_(param, non_blocking=True)
        self.storage_size = param.storage().size()
        self.depth = 0

    def swap_in(self):
        """Re-materialize the device storage and refill it from the mirror."""
        if self.param.storage().size() == 0:
            self.param.storage().resize_(self.storage_size)
            # '.data' copy: does not bump the version counter (see module doc)
            self.param.data.copy_(self.cpu, non_blocking=True)

    def swap_out(self, copy_to_host=False):
        """Release the device storage; no-op if already released.

        ``copy_to_host=True`` refreshes the mirror first (used after the
        tensor was modified on device).
        """
        if self.param.storage().size() != 0:
            if copy_to_host:
                self.cpu.copy_(self.param, non_blocking=True)
            self.param.storage().resize_(0)

    def acquire(self):
        """Enter a usage scope: materialize and bump the nesting depth."""
        self.depth += 1
        self.swap_in()

    def release(self, copy_to_host=False):
        """Exit a usage scope: release only when the outermost scope exits.

        The depth floor keeps the counter self-healing if a release ever sees
        an already-released tensor (swap_out is idempotent).
        """
        if self.depth > 0:
            self.depth -= 1
        if self.depth == 0:
            self.swap_out(copy_to_host=copy_to_host)


class SwapManager:
    """Registry of all swap states of the current process."""

    def __init__(self):
        self.states = []

    def reset(self):
        """Drop every registry entry; states stay attached to their tensors
        and are re-adopted lazily on next use.
        """
        self.states = []

    def register(self, state):
        if state not in self.states:  # identity-based; idempotent
            self.states.append(state)

    def swap_all_in(self):
        for state in self.states:
            state.swap_in()

    def swap_all_out(self, copy_to_host=False):
        """Release every state; refuses to run inside an open block_swap
        scope (see release_states_after_update below).
        """
        release_states_after_update(self.states, copy_to_host=copy_to_host)

    @property
    def total_numel(self):
        return sum(state.param.numel() for state in self.states)


_MANAGER = SwapManager()


def get_swap_manager():
    """Return the process-wide swap manager."""
    return _MANAGER


def release_states_after_update(states, copy_to_host=True):
    """Exit an optimizer update window: refresh mirrors and release.

    Refuses to release a tensor whose scope is still open: an optimizer step
    inside ``with block_swap(...)`` would release storage the block still uses.
    """
    for state in states:
        if state.depth != 0:
            raise RuntimeError(
                '[block-swap] optimizer update ran inside an open block_swap '
                'scope (tensor shape '
                f'{tuple(state.param.shape)}, dtype {state.param.dtype}); move '
                'the optimizer step outside the with block.'
            )
    for state in states:
        # tensors were updated on device: refresh mirrors (D2H) + release
        state.swap_out(copy_to_host=copy_to_host)
