# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pylint: skip-file

import warnings
from functools import wraps
from typing import List

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.distributed.param_and_grad_buffer import (
    dist_all_gather_func,
    shard_buffer,
)


class _LayerwiseAllGatherHandle:
    """Handle wrapping multiple async all-gather work objects.

    Megatron native waits on only the last handle for in-order completion on
    the same communicator.
    """

    def __init__(self, handles):
        self.handles = handles

    def wait(self):
        """Wait on the last handle and clear all handles."""
        if self.handles:
            self.handles[-1].wait()
        self.handles = None


def set_layerwise_params_list(self, layerwise_params_list: List[List[torch.nn.Parameter]]):
    """Set per-rank parameter lists for layer-wise async all-gather.

    Args:
        layerwise_params_list: List of param lists, one per rank in the DP group.
            Each inner list contains the parameters owned by that rank's
            layer-wise optimizer that also belong to this bucket.
    """
    self.layerwise_params_list = layerwise_params_list
    self.layerwise_param_flat_sizes = [sum([p.numel() for p in param_list]) for param_list in layerwise_params_list]
    self.layerwise_gather_list = None


def param_and_grad_bucket_group_init_wrapper(init_func):
    @wraps(init_func)
    def wrapper(self, buckets, ddp_config, collective_group, collective_group_size, *args, **kwargs):
        init_func(self, buckets, ddp_config, collective_group, collective_group_size, *args, **kwargs)
        # overlap_param_gather covers the layer-wise optimizer case, which sets
        # overlap_param_gather=True without use_distributed_optimizer.
        if ddp_config.overlap_param_gather and not hasattr(self, "intra_distributed_optimizer_instance_group"):
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = torch.distributed.get_rank(group=collective_group)
        if ddp_config.overlap_param_gather and not hasattr(self, "data_parallel_group"):
            self.data_parallel_group = collective_group

    return wrapper


def distributed_data_parallel_start_param_sync_wrapper(func):
    @wraps(func)
    def wrapper(
        self,
        *unused,
        force_sync: bool = False,
        force_dispatch: bool = False,
        dense_or_moe_group: str = None,
    ):
        ddp_config = getattr(self, "ddp_config", None)
        if not (
            getattr(ddp_config, "overlap_param_gather", False)
            and not getattr(ddp_config, "use_distributed_optimizer", False)
        ):
            return func(
                self,
                *unused,
                force_sync=force_sync,
                force_dispatch=force_dispatch,
                dense_or_moe_group=dense_or_moe_group,
            )

        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if getattr(self, "overlap_param_gather_with_optimizer_step", False) and not force_dispatch:
                return

        if dense_or_moe_group is None:
            bucket_groups = self.bucket_groups + self.expert_parallel_bucket_groups
        elif dense_or_moe_group == "dense":
            bucket_groups = self.bucket_groups
        elif dense_or_moe_group == "moe":
            bucket_groups = self.expert_parallel_bucket_groups
        else:
            raise ValueError(f"Invalid dense_or_moe_group: {dense_or_moe_group}")

        for bucket_group in bucket_groups:
            if not force_sync and (
                getattr(bucket_group, "param_gather_handle", None) is not None
                or getattr(bucket_group, "param_gather_dispatched", False)
            ):
                continue
            bucket_group.start_param_sync(force_sync=force_sync)

    return wrapper


def start_param_sync(self, force_sync: bool = False):
    """
    Initiates all necessary param all-gathers for this bucket.

    When ddp_config.overlap_param_gather is set to True, dispatches an asynchronous
    communication call (unless force_sync is True). When ddp_config.overlap_param_gather
    is set to False, makes synchronous call.

    Args:
        force_sync (bool, optional): force synchronous collective regardless of
            other settings if true.
    """
    # overlap_param_gather covers the layer-wise optimizer case, which sets
    # overlap_param_gather=True without use_distributed_optimizer.
    if not (self.ddp_config.use_distributed_optimizer or self.ddp_config.overlap_param_gather):
        raise ValueError("Either use_distributed_optimizer or overlap_param_gather must be True")

    if force_sync:
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            return
    else:
        if self.param_gather_handle is not None:
            raise ValueError("param_gather_handle should be None when not force_sync")

    async_op = self.ddp_config.overlap_param_gather and not force_sync

    if not self.ddp_config.use_distributed_optimizer:
        # Layer-wise optimizer path: use all_gather for variable-size
        # param gather.
        #
        # Each rank may own a different number of params per bucket, so
        # layerwise_param_flat_sizes can vary across ranks.  PyTorch's NCCL
        # backend handles uneven tensor sizes in torch.distributed.all_gather
        # (falling back to grouped send/recv internally when sizes differ),
        # so no manual padding is needed.
        dp_size = self.intra_distributed_optimizer_instance_size
        if dp_size == 1:
            # Single-rank group (e.g., expt_dp_size == 1): no all-gather needed.
            self.param_gather_dispatched = True
            return

        local_rank = self.intra_distributed_optimizer_instance_rank
        group = self.intra_distributed_optimizer_instance_group
        layerwise_work_handles = []
        for bucket in self.buckets:
            if getattr(bucket, "layerwise_param_flat_sizes", None) is None:
                continue
            # Use param dtype (e.g., bf16), NOT grad dtype (which may be
            # fp32 when grad_reduce_in_fp32 is enabled).
            param_dtype = bucket.params_list[0].dtype
            if max(bucket.layerwise_param_flat_sizes) == 0:
                bucket.layerwise_gather_list = None
                continue

            local_size = bucket.layerwise_param_flat_sizes[local_rank]
            total_gather_size = sum(bucket.layerwise_param_flat_sizes)

            # Reuse grad_data as the all_gather receive buffer; it is idle
            # during forward and grad_dtype.element_size >= param_dtype.
            reuse_buf = bucket.grad_data.view(param_dtype)
            if reuse_buf.numel() < total_gather_size:
                raise ValueError(f"grad_data buffer too small: {reuse_buf.numel()} < {total_gather_size}")

            # Partition reuse_buf into contiguous per-rank receive slices.
            gather_list = []
            offset = 0
            for i in range(dp_size):
                size = bucket.layerwise_param_flat_sizes[i]
                gather_list.append(reuse_buf[offset : offset + size])
                offset += size
            local_slot_view = gather_list[local_rank]

            # Flatten local params and copy into the local rank's slot.
            # Detach from autograd since start_param_sync may be called
            # during the forward pass where autograd is active.
            if local_size > 0:
                flat_local_params = _flatten_dense_tensors(bucket.layerwise_params_list[local_rank]).detach()
                local_slot_view.copy_(flat_local_params)
            bucket.layerwise_gather_list = gather_list

            work = torch.distributed.all_gather(gather_list, local_slot_view, group=group, async_op=async_op)
            if async_op and work is not None:
                layerwise_work_handles.append(work)

        if async_op:
            self.param_gather_handle = _LayerwiseAllGatherHandle(layerwise_work_handles)
        else:
            # Synchronous: unflatten and copy gathered params immediately.
            for bucket in self.buckets:
                if bucket.layerwise_gather_list is None:
                    continue
                for idx, params in enumerate(bucket.layerwise_params_list):
                    if len(params) == 0 or idx == local_rank:
                        continue
                    updated_params = _unflatten_dense_tensors(bucket.layerwise_gather_list[idx], params)
                    for updated_p, model_p in zip(updated_params, params):
                        model_p.data.copy_(updated_p)
                bucket.layerwise_gather_list = None
            self.param_gather_handle = None
    else:
        self.param_gather_handle = []
        for bucket in self.buckets:
            local_data_view = shard_buffer(bucket.param_data, self.intra_distributed_optimizer_instance_size)[
                self.intra_distributed_optimizer_instance_rank
            ]
            handle = dist_all_gather_func(
                bucket.param_data,
                local_data_view,
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op,
            )
            self.param_gather_handle.append(handle)
        if not async_op:
            self.param_gather_handle = None
    self.param_gather_dispatched = True


def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
    """
    Finishes param sync communication operation for this bucket. Dispatches
    next bucket's param sync if available, unless skip_next_bucket_dispatch
    is True.

    When ddp_config.overlap_param_gather is set to True, waits for asynchronous
    communication call to complete (and dispatches one if one is not already
    outstanding). Throws assertion error if ddp_config.overlap_param_gather is set to
    False.

    Args:
        skip_next_bucket_dispatch (bool, optional): if true, dispatch next
            bucket's communication if available.
    """
    if not self.ddp_config.overlap_param_gather:
        raise ValueError("overlap_param_gather must be True")

    # If current bucket's param AG has not been dispatched, dispatch it now (e.g., first
    # AG bucket in first model chunk if ddp_config.align_param_gather is False).
    if not self.param_gather_dispatched:
        self.start_param_sync()

    if self.param_gather_handle is not None:
        self.param_gather_handle.wait()
        self.param_gather_handle = None
        # Dispatch next bucket's asynchronous param AG.
        if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
            if self.next_param_gather_bucket_group.param_gather_dispatched:
                warnings.warn(
                    "The next bucket's parameter all-gather operation has already been "
                    "dispatched. This may be caused by a mismatch between the order of "
                    "parameter registration and forward pass execution, which will "
                    "hurt the communication-computation overlap performance."
                )
            else:
                self.next_param_gather_bucket_group.start_param_sync()

        if not self.ddp_config.use_distributed_optimizer:
            for bucket in self.buckets:
                if bucket.layerwise_gather_list is None:
                    continue
                # Unflatten and copy gathered params for each rank.
                for idx, params in enumerate(bucket.layerwise_params_list):
                    # Skip local params and empty tensors.
                    if len(params) == 0 or idx == self.intra_distributed_optimizer_instance_rank:
                        continue
                    updated_params = _unflatten_dense_tensors(bucket.layerwise_gather_list[idx], params)
                    for updated_p, model_p in zip(updated_params, params):
                        model_p.data.copy_(updated_p)
                bucket.layerwise_gather_list = None


def finish_grad_sync_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.param_gather_dispatched = False
        return func(self, *args, **kwargs)

    return wrapper
