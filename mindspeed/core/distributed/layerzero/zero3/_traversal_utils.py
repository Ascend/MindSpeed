"""
NOTE: This file must be imported like
``import torch.distributed.fsdp._traversal_utils`` and not like
``from torch.distirbuted.fsdp._traversal_utils import ...`` to avoid circular
imports. For brevity, we may import the file as ``traversal_utils``.
"""

import collections
from typing import Deque, List, Set, Tuple, TYPE_CHECKING

import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from mindspeed.core.distributed.layerzero.zero3._common_utils import _get_module_zero3_state

if TYPE_CHECKING:
    from mindspeed.core.distributed.layerzero.zero3._common_utils import _ZeRO3State

"""
[Note: ZeRO3 State Traversal]
For the wrapper code path, ``_ZeRO3PState`` is the ``ZeRO3``
module wrapping a fully sharded module, and for the non-wrapper code path,
``_ZeRO3PState`` is an object that gets embedded on a fully sharded module.

There are three common traversal idioms: Given a root module,
- ``_get_zero3_states()`` returns all ``_ZeRO3PState`` s in the tree.
- ``get_zero3_root_states()`` returns all local root ``_ZeRO3PState`` s in the
tree (i.e. those with ``_is_root == True``).
- ``_get_zero3_handles()``returns all ``FlatParamHandle`` s in the tree.

All of these methods must take in the root module (i.e. an ``nn.Module``) and
not a general ``_ZeRO3PState`` because ``_ZeRO3PState`` does not support a graph
traversal, whereas ``nn.Module`` has ``nn.Module.modules()`` for traversal.
"""


def _composable(module: nn.Module) -> bool:
    """
    Returns if ``module`` can compose with ``fully_shard``.
    """
    return "replicate" not in _get_registry(module)


def _get_zero3_states_with_modules(
    module: nn.Module,
) -> Tuple[List["_ZeRO3State"], List[nn.Module]]:
    """
    Returns a tuple containing:
    1. A list of the ``"_ZeRO3State"`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the modules owning the states in the first list.

    For the wrapper code path, both returned lists are the same, each
    containing all ``FullyShardedDataParallel`` instances. For the composable
    code path, this returns a list of all composable state instances and a list
    of the corresponding fully sharded modules. See [Note: Fully Sharded
    Module].

    NOTE: The traversal does not proceed into any module annotated by an
    incompatible API (e.g. ``replicate``).
    """
    zero3_states: List["_ZeRO3State"] = []
    zero3_modules: List[nn.Module] = []
    # Track the visited FSDP states since multiple modules may share the same
    # one and we want to return a de-duplicated list
    visited_states: Set["_ZeRO3State"] = set()
    # Track the visited modules in case of shared modules, which implies the
    # module graph is no longer a tree
    visited_modules: Set[nn.Module] = set()

    # Perform depth-first search from `module` to ensure that we do not
    # traverse into an incompatible API's subtree (use DFS instead of BFS to
    # match `.modules()` order)
    deque: Deque[nn.Module] = collections.deque([module])
    while deque:
        submodule = deque.popleft()
        visited_modules.add(submodule)
        if not _composable(submodule):
            continue
        for child_module in reversed(list(submodule.children())):
            if child_module not in visited_modules:
                deque.appendleft(child_module)
        optional_state = _get_module_zero3_state(submodule)
        if optional_state is not None and optional_state not in visited_states:
            visited_states.add(optional_state)
            zero3_states.append(optional_state)
            zero3_modules.append(submodule)
    return zero3_states, zero3_modules


def _get_zero3_states(module: nn.Module) -> List["_ZeRO3State"]:
    """See :func:`_get_zero3_states_with_modules`."""
    zero3_states, _ = _get_zero3_states_with_modules(module)
    return zero3_states


def _get_zero3_handles(module: nn.Module) -> List:
    """
    Returns all ``FlatParamHandle`` s in the module tree rooted at ``module``
    following the rules in :func:`_get_zero3_state`.
    """
    handles = [
        zero3_state._handle
        for zero3_state in _get_zero3_states(module)
        if zero3_state._handle is not None
    ]
    return handles
