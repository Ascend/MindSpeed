import logging
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.nn as nn
from mindspeed.core.distributed.layerzero.zero3.flat_param import FlatParamHandle
import mindspeed.core.distributed.layerzero.zero3._traversal_utils as traversal_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""

    NONE = auto()  # no deviation yet
    WARNING = auto()  # deviated this iteration; currently issuing warnings
    WARNED = auto()  # deviated in a previous iteration


class _ExecOrderData:

    def __init__(
        self,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None:
        # Tracks the (static) pre-forward order for execution order validation
        # and forward prefetching
        self.handles_pre_forward_order: List[FlatParamHandle] = []
        # Tracks the post-forward order for pre-backward prefetching
        self.handles_post_forward_order: List[Optional[FlatParamHandle]] = []
        self._iter = 0

        # Gives the max number of backward/forward prefetched all-gathers by a
        # single module
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        self.process_group: Optional[dist.ProcessGroup] = None
        self.world_size: Optional[int] = None
        self.all_handles: List[FlatParamHandle] = []

    def init(
        self,
        state,
        root_module: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> None:
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        # Fix an order over the handles, which should be the same across ranks
        for handle in traversal_utils._get_zero3_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index

    @property
    def is_first_iter(self) -> bool:
        return self._iter == 0

    def get_handle_to_backward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> Optional[FlatParamHandle]:
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(
        self,
        current_handle: FlatParamHandle,
    ) -> Optional[FlatParamHandle]:
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle: Optional[FlatParamHandle] = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle

    def get_handle_to_post_backward(
        self,
        current_handle: FlatParamHandle,
    ) -> List[FlatParamHandle]:
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return []
        target_index = current_index + 1
        target_handle: List[FlatParamHandle] = []
        for _ in range(len(self.handles_pre_forward_order)):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle.append(self.handles_pre_forward_order[target_index])
            target_index += 1
        return target_handle

    def record_post_forward(self, handle: Optional[FlatParamHandle]) -> None:
        if not handle or handle._post_forward_index is not None:
            return
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index

        self.handles_post_forward_order.append(handle)

    def record_pre_forward(
        self, handle: Optional[FlatParamHandle], is_training: bool
    ) -> None:
        if not handle:
            return
        # Fix the order after the first iteration and only record the first
        # usage of a handles key
        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        self.handles_pre_forward_order.append(handle)

    def next_iter(self):
        self._iter += 1
        self.handles_post_forward_order.clear()

    def next_iter_during_accumulation(self):
        self._iter += 1
