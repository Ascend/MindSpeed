import abc
import threading
from typing import (
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch.utils.hooks import RemovableHandle
from torch.autograd.graph import Node


class _MultiHandle(RemovableHandle):
    handles: Tuple[RemovableHandle, ...]

    def __init__(self, handles: Tuple[RemovableHandle, ...]) -> None:
        self.handles = handles

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __getstate__(self) -> Tuple[RemovableHandle, ...]:
        return self.handles

    def __setstate__(self, state: Tuple[RemovableHandle, ...]) -> None:
        self.handles = state


def _get_grad_fn_or_grad_acc(t: Union[torch.Tensor, None]) -> Node:

    if not (isinstance(t, torch.Tensor) and t.requires_grad):
        raise ValueError(
            f"Expects torch.Tensor with requires_grad=True, got {type(t)}")
    if t.requires_grad and t.grad_fn is not None:
        node = t.grad_fn
    else:
        with torch.enable_grad():
            node = t.grad_fn.next_functions[0][0]
    if node is None:
        raise AssertionError(
            f"No graph.Node object returned from tensor.grad_fn")
    return node


def register_multi_post_grad_hook(
    tensors: Sequence[torch.Tensor],
    fn: Union[
        Callable[[Sequence[Optional[torch.Tensor]]], None],
        Callable[[torch.Tensor], None],
    ],
) -> RemovableHandle:
    """Note: 
    1. This hook is only called once, so it needs to be re-registered.
    2. This hook is called only when all grad_fn or acc node is triggered
    """
    lock = threading.Lock()
    nb_calls = 0
    grad_fns = list(map(_get_grad_fn_or_grad_acc, tensors))
    len_tensors = len(tensors)

    def get_inner_hook() -> Callable[[torch.Tensor], None]:
        def inner_hook(*grad: torch.Tensor) -> None:
            nonlocal len_tensors, nb_calls, fn
            with lock:
                nb_calls += 1
                if len_tensors == nb_calls:
                    fn()
        return inner_hook

    handles = tuple(
        t.register_hook(get_inner_hook()) for t in grad_fns
    )
    return _MultiHandle(handles)
