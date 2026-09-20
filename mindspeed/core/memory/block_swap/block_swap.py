# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""Block-level CPU-offload context manager.

Usage::

    from mindspeed.core.memory.block_swap import block_swap

    with block_swap(self.weight1, self.weight2):
        hidden = torch.bmm(x, self.weight1.transpose(-1, -2))
        out = torch.bmm(hidden, self.weight2)
    # exiting the block releases both weights' device storage

Semantics: enter materializes the tensors (H2D from a pinned mirror, created
lazily on first use); exit releases the storage. Nesting is depth-counted
(only the outermost exit releases). Blocks are training-safe: tensors saved
via ``save_for_backward`` inside the block are re-materialized when a later
backward unpacks them (saved-tensor hooks), so backward may run outside the
block. Megatron optimizer steps over adopted tensors are materialized and
refreshed automatically (see megatron_integration.py) and must not run
inside a block.
"""

import torch

from mindspeed.core.memory.block_swap.megatron_integration import (
    ensure_megatron_optimizer_integration,
)
from mindspeed.core.memory.block_swap.swap_state import (
    SwapState,
    get_swap_manager,
)


def _validate_swap_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'block_swap expects torch tensors, got {type(tensor).__name__}')
    if tensor.device.type == 'cpu':
        raise ValueError(
            'block_swap manages device-side storage; the tensor is on CPU already. '
            'Move it to the device first (model weights: build/move the model '
            'before wrapping).'
        )
    # a view would release its siblings too (resize acts on the whole storage)
    storage = tensor.untyped_storage()
    if storage.nbytes() != 0:
        own_bytes = tensor.numel() * tensor.element_size()
        if tensor.storage_offset() != 0 or own_bytes != storage.nbytes():
            import warnings

            warnings.warn(
                'block_swap: tensor is a view into a larger storage; releasing it '
                'will also release sibling tensors sharing that storage.',
                stacklevel=3,
            )


class block_swap:
    """Block-level cpu-offload scope for arbitrary tensors.

    Args:
        *tensors: device tensors to manage inside the block (Parameters or
            plain tensors).
        copy_back: refresh the pinned mirrors from the device values on exit
            (needed only for in-place writes through ``.data``; version-bumping
            in-place ops are detected automatically). Default ``False`` keeps
            the exit copy-free.
    """

    def __init__(self, *tensors, copy_back=False):
        if not tensors:
            raise ValueError('block_swap expects at least one tensor')
        for tensor in tensors:
            _validate_swap_tensor(tensor)
        self.tensors = list(tensors)
        self.copy_back = copy_back
        self.states = []
        self._entry_versions = None
        self._hooks_ctx = None

    # -- context manager protocol -------------------------------------------

    def __enter__(self):
        manager = get_swap_manager()
        # adopting tensors implies Megatron optimizer updates may touch them:
        # install the integration wrappers (idempotent, lazy Megatron import)
        ensure_megatron_optimizer_integration()
        self.states = []
        self._entry_versions = [getattr(t, '_version', None) for t in self.tensors]
        for tensor in self.tensors:
            state = getattr(tensor, 'block_swap_state', None)
            if state is None:
                # lazy adoption: mirror the current device values
                state = SwapState(tensor)
                tensor.block_swap_state = state
            # idempotent (identity-based): re-adopts states that survived a
            # manager reset and keeps optimizer updates covering them
            manager.register(state)
            self.states.append(state)
            state.acquire()
        # re-materialize managed tensors when a later backward unpacks them
        self._hooks_ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # leave the hook scope first: future saves are not intercepted
        self._hooks_ctx.__exit__(exc_type, exc_value, traceback)
        for tensor, state, entry_version in zip(self.tensors, self.states, self._entry_versions):
            copy_back = self.copy_back
            if not copy_back and entry_version is not None and getattr(tensor, '_version', None) != entry_version:
                # in-place ops bumped the version counter: the mirror is stale.
                # Writes through .data do not bump it and stay undetectable.
                import warnings

                warnings.warn(
                    'block_swap: tensor was modified in place inside the block '
                    '(version counter changed); refreshing the pinned mirror '
                    'automatically. Writes through .data are not detectable - '
                    'pass copy_back=True for those.',
                    stacklevel=2,
                )
                copy_back = True
            state.release(copy_to_host=copy_back)
        self.states = []
        self._entry_versions = None
        return False

    # -- saved-tensor hooks ---------------------------------------------------

    @staticmethod
    def _pack(tensor):
        # keep the tensor reference unchanged; unpack carries the swap-in
        return tensor

    @staticmethod
    def _unpack(saved):
        # raw swap_in (not acquire): the block scope has ended, backward owns
        # the tensor now; it stays materialized until the next release point
        state = _state_of(saved)
        if state is not None:
            state.swap_in()
        return saved


def _state_of(tensor):
    """Swap state of ``tensor`` or of the managed tensor it is a view of.

    Ops may save a *view* of a managed tensor (e.g. ``matmul(x, w.t())`` saves
    the transposed view); the state lives on the base tensor, so walk the
    autograd view chain (``_base``).
    """
    node = tensor
    while node is not None:
        state = getattr(node, 'block_swap_state', None)
        if state is not None:
            return state
        node = getattr(node, '_base', None)
    return None


# -- manual bulk control for custom training loops ----------------------------


def block_swap_all_in():
    """Materialize every managed tensor (e.g. before a manual optimizer update)."""
    get_swap_manager().swap_all_in()


def block_swap_all_out(copy_to_host=True):
    """Release every managed tensor (e.g. after a manual optimizer update)."""
    get_swap_manager().swap_all_out(copy_to_host=copy_to_host)
