# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import math
from contextlib import nullcontext
from functools import wraps
from logging import getLogger
from typing import Dict, List

import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import (
    BufferType,
    dist_all_gather_func,
    dist_reduce_scatter_func,
    shard_buffer,
)
from megatron.core.utils import (
    is_torch_min_version,
    log_on_each_pipeline_stage,
)
from megatron.core.fp8_utils import is_float8tensor
from megatron.training import get_args


logger = getLogger(__name__)


def pipe_register_grad_ready_wrapper(register_grad_ready):
    @wraps(register_grad_ready)
    def wrapper(self, param: torch.nn.Parameter):
        assert (self.ddp_config.overlap_grad_reduce), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        from mindspeed.moe.pipe_experts import FLAG_GRAD_REDUCE
        if self.is_last_microbatch and FLAG_GRAD_REDUCE:
            register_grad_ready(self, param)

    return wrapper


def reuse_fp32_param_param_and_grad_buffer_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_param_and_grad_buffer_init(*args, **kwargs):
        global_args = get_args()
        math_ceil = math.ceil
        if global_args.reuse_fp32_param and global_args.use_distributed_optimizer:
            def ceil_even(x):
                return math_ceil(math_ceil(x) / 2) * 2
            math.ceil = ceil_even
        init_func(*args, **kwargs)
        if global_args.reuse_fp32_param and global_args.use_distributed_optimizer:
            math.ceil = math_ceil
    return reuse_fp32_param_param_and_grad_buffer_init


# The patch is a temporary patch and can be removed once PTA supports the _coalescing_manager capability.
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
    assert self.ddp_config.use_distributed_optimizer

    if force_sync:
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            return
    else:
        assert self.param_gather_handle is None

    async_op = self.ddp_config.overlap_param_gather and not force_sync

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


# The patch is a temporary patch and can be removed once PTA supports the _coalescing_manager capability.
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
    assert self.ddp_config.use_distributed_optimizer
    assert self.ddp_config.overlap_param_gather

    # If current bucket's param AG has not been dispatched, dispatch it now (e.g., first
    # AG bucket in first model chunk if ddp_config.align_param_gather is False).
    if not self.param_gather_dispatched:
        self.start_param_sync()

    if self.param_gather_handle is not None:
        for handle in self.param_gather_handle:
            handle.wait()
        self.param_gather_handle = None
        # Dispatch next bucket's asynchronous param AG.
        if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
            self.next_param_gather_bucket_group.start_param_sync()


# The patch is a temporary patch and can be removed once PTA supports the _coalescing_manager capability.
def start_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the bucket group.

    When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
    communication call. When ddp_config.overlap_grad_reduce is set to False, makes
    synchronous call.
    """
    assert (
        self.grad_reduce_handle is None
    ), 'Should not have multiple communication calls outstanding at once'

    if self.ddp_config.check_for_nan_in_grad:
        self.check_grads(
            check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
            check_for_large=self.ddp_config.check_for_large_grads,
        )

    # gradient_scaling_factor already takes into account whether we are computing
    # an average or sum in the data-parallel collective.
    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor

    # Decide reduce_op.
    reduce_op = torch.distributed.ReduceOp.SUM
    if self.ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG

    # Use async communications only when overlap_grad_reduce is True.
    async_op = (
        self.ddp_config.overlap_grad_reduce
        and self.ddp_config.num_distributed_optimizer_instances == 1
    )
    if (
        self.ddp_config.num_distributed_optimizer_instances > 1
        and self.ddp_config.overlap_grad_reduce
    ):
        # Assign a communication stream if we use partial DP DistOpt and we
        # need to overlap communication
        stream_context = torch.cuda.stream(self.communication_stream)

        # The RS/AR communication stream needs to wait for the default stream
        # to complete its gradient computation before launching the next
        # gradient reduction collective
        self.communication_stream.wait_stream(torch.cuda.default_stream())
    else:
        stream_context = nullcontext()

    if self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    # Coalesce communication kernels across buckets in the bucket group.
    self.grad_reduce_handle = []
    for bucket in self.buckets:
        if self.ddp_config.use_distributed_optimizer:
            local_data_view = shard_buffer(bucket.grad_data, self.intra_distributed_optimizer_instance_size)[
                self.intra_distributed_optimizer_instance_rank
            ]
            handle = dist_reduce_scatter_func(
                local_data_view,
                bucket.grad_data,
                op=reduce_op,
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op,
            )
        else:
            handle = torch.distributed.all_reduce(
                bucket.grad_data,
                op=reduce_op,
                group=self.data_parallel_group,
                async_op=async_op,
            )
        self.grad_reduce_handle.append(handle)

    # When enabling partial DP domain DistOpt, we need to All-Reduce across all partial domains
    if (
        self.ddp_config.use_distributed_optimizer
        and self.ddp_config.num_distributed_optimizer_instances > 1
    ):
        self.grad_reduce_handle = []
        # Create a new coalescing facility for the inter partial DP-AllReduce here
        for bucket in self.buckets:
            if self.ddp_config.use_distributed_optimizer:
                local_data_view = shard_buffer(bucket.grad_data, self.intra_distributed_optimizer_instance_size)[
                    self.intra_distributed_optimizer_instance_rank
                ]
                handle = torch.distributed.all_reduce(
                    local_data_view,
                    op=reduce_op,
                    group=self.inter_distributed_optimizer_instance_group,
                    async_op=async_op,
                )
            self.grad_reduce_handle.append(handle)
    if not async_op:
        self.grad_reduce_handle = None


# The patch is a temporary patch and can be removed once PTA supports the _coalescing_manager capability.
def finish_grad_sync(self):
    """
    Finishes grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the bucket group.

    When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
    communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
    makes synchronous call.
    """
    # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
    self.param_gather_dispatched = False
    if not self.ddp_config.overlap_grad_reduce:
        self.start_grad_sync()
        return
    # When using partial DP DistOpt, we don't need to sync as we launch comms on a separate
    # communication stream
    if self.ddp_config.num_distributed_optimizer_instances > 1:
        torch.cuda.default_stream().wait_stream(self.communication_stream)
        return
    assert self.grad_reduce_handle is not None, (
        f'Communication call has not been issued for this bucket '
        f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
    )
    for handle in self.grad_reduce_handle:
        handle.wait()
    self.grad_reduce_handle = None


def param_and_grad_buffer_init_pad(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
):
    self.ddp_config = ddp_config
    self.params = params
    self.param_indices = param_indices

    # Check that params are unique.
    unique_params = set()
    for param in params:
        assert param not in unique_params
        unique_params.add(param)
    del unique_params

    # Store attributes that will be needed later.
    self.param_dtype = param_dtype
    self.grad_dtype = grad_dtype
    self.data_parallel_group = data_parallel_group
    self.data_parallel_world_size = torch.distributed.get_world_size(
        group=self.data_parallel_group
    )
    self.gradient_scaling_factor = gradient_scaling_factor

    # Data structures to store underlying buckets and relevant indexing data.
    self.buckets = []
    self.param_to_bucket = {}  # Param -> bucket mapping.
    self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

    def _pad(number_to_be_padded: int, divisor: int) -> int:
        return int(math.ceil(number_to_be_padded / divisor) * divisor)

    def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
        """
        Pads end index of bucket if using distributed optimizer (to ensure uniform sharding).
        """
        if self.ddp_config.use_distributed_optimizer:
            # We now ensure that all buckets start at a memory address that is 512-byte
            # If using a distributed optimizer, pad the memory buffer to be
            # multiple of data_parallel_world_size. (This padding is done
            # due to a constraint with the reduce_scatter op, which requires
            # all tensors have equal size.)
            # 512-byte for Ascend, 256-byte for nv.

            element_size = 4 if param_dtype == torch.float else 2
            global_args = get_args()
            align_size = global_args.param_and_grad_buffer_pad // element_size
            return _pad(bucket_end_index, self.data_parallel_world_size * align_size)
        return bucket_end_index

    def _pad_start_of_param_if_needed(param_start_index: int) -> int:
        """
        Pads start index of param if using distributed optimizer (to ensure "good" alignment).
        """
        if self.ddp_config.use_distributed_optimizer:
            # Ensure that params start at 128-byte aligned addresses (64 values
            # since params are >= 16-bit precision).
            return _pad(param_start_index, 64)
        return param_start_index

    # First, figure out how many elements should be in the underlying buffer storage.
    # Note that if we need to split the buffer into smaller buckets, each of these
    # might need to be padded as well (if using the distributed optimizer).
    param_start_index = 0
    bucket_start_index = param_start_index
    bucket_params = set()
    self.bucket_indices = []
    per_bucket_numel_unpadded = []
    bucket_id = 0

    def _update_bucket_metadata(param_end_index: int) -> int:
        """
        Record metadata for the bucket starting at bucket_start_index and ending with the
        passed-in param_end_index. Returns the bucket's end_index.
        """
        nonlocal bucket_start_index, bucket_params, bucket_id
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)

        # Record metadata of new bucket.
        self.bucket_indices.append((bucket_start_index, bucket_end_index))
        bucket_start_index = bucket_end_index

        # Prepare for next bucket.
        bucket_params = set()
        bucket_id += 1

        # Return the potentially padded bucket_end_index.
        return bucket_end_index

    def _does_param_require_new_bucket(param):
        """
        Split shared embedding parameters into separate bucket if using distributed
        optimizer that makes use of reduce-scatters instead of all-reduces.
        This ensures that the first and last pipeline stage partition optimizer state
        for the shared embedding parameters the same way across DP replicas, allowing
        the DP reduce-scatter to be before the embedding all-reduce.
        """
        return (
                getattr(param, "shared_embedding", False)
                and self.ddp_config.use_distributed_optimizer
        )

    for param in params[::-1]:
        # Iterate through parameters in reverse order to roughly follow backprop order.

        this_numel = param.data.nelement()
        param_start_index = _pad_start_of_param_if_needed(param_start_index)

        # Create bucket with collected parameters if current param needs its own bucket.
        if _does_param_require_new_bucket(param):
            # We are creating a bucket for the already accumulated parameters, whose params
            # end at the current param_start_index.
            if self.ddp_config.use_distributed_optimizer:
                # Make sure new bucket is appropriately padded.
                if param_start_index % self.data_parallel_world_size != 0:
                    param_start_index = _pad_end_of_bucket_if_needed(param_start_index)
            if len(bucket_params) > 0:
                bucket_end_index = _update_bucket_metadata(param_start_index)

        param_end_index = param_start_index + this_numel
        self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        # If we have enough elements already or the current param is part of the shared
        # embedding layer and needs a separate bucket, form a new bucket.
        if (
                bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
        ) or _does_param_require_new_bucket(param):
            bucket_end_index = _update_bucket_metadata(param_end_index)
            param_start_index = bucket_end_index
        else:
            param_start_index = param_end_index

    # Add remaining params to a new bucket.
    if len(bucket_params) > 0:
        bucket_end_index = _update_bucket_metadata(param_end_index)

    # Next, create underlying storage for buffer (with numel elements that includes
    # padding as necessary).
    self.numel = bucket_end_index
    self.numel_unpadded = sum(per_bucket_numel_unpadded)
    assert self.numel_unpadded <= self.numel
    if self.ddp_config.use_distributed_optimizer:
        assert self.numel % self.data_parallel_world_size == 0
    else:
        assert self.numel == self.numel_unpadded

    self.param_data = None
    # Only re-map param tensors if using distributed optimizer.
    if self.ddp_config.use_distributed_optimizer:
        self.param_data = torch.zeros(
            self.numel,
            dtype=self.param_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
    self.grad_data = torch.zeros(
        self.numel,
        dtype=self.grad_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    # Finally, map param.data and param.main_grad fields to buffers.
    bucket_params = []
    bucket_start_index = 0
    cur_bucket_id = 0
    for param in params[::-1]:
        param_start_index, param_end_index, bucket_id = self.param_index_map[param]

        # Assign param.data to appropriate segment of self.param_data.
        if self.param_data is not None:
            old_param_data = param.data
            new_param_data = self._get(
                param.data.shape, param_start_index, buffer_type=BufferType.PARAM
            )
            if is_float8tensor(param):
                param._data = new_param_data
            else:
                param.data = new_param_data
            assert old_param_data._base is None
            # Copy tensor values (from initialization or checkpoint).
            param.data.detach().copy_(old_param_data)
            del old_param_data

        param.main_grad = self._get(
            param.data.shape, param_start_index, buffer_type=BufferType.GRAD
        )
        if bucket_id != cur_bucket_id:
            bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
            self.buckets.append(
                self._new_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_start_index,
                    end_index=bucket_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
            )
            bucket_start_index = bucket_end_index
            bucket_params = []
            assert cur_bucket_id + 1 == len(self.buckets)
            assert bucket_id == cur_bucket_id + 1
            cur_bucket_id = bucket_id
        bucket_params.append(param)

    # Add remaining params to a new bucket.
    if len(bucket_params) > 0:
        bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
        self.buckets.append(
            self._new_bucket(
                bucket_params=bucket_params,
                start_index=bucket_start_index,
                end_index=bucket_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )
        )

    # Log buckets for all PP stages.
    log_strs = []
    log_strs.append(
        f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
    )
    for index, bucket in enumerate(self.buckets):
        numel = 0
        for param in bucket.params:
            numel += param.data.nelement()
        log_strs.append(f'Params for bucket {index + 1} ({numel} elements):')
        for param in bucket.params:
            log_strs.append(f'\t{param_to_name[param]}')
    log_on_each_pipeline_stage(logger, logging.INFO, '\n'.join(log_strs))
