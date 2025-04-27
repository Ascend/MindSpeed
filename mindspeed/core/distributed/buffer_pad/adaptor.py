# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import Dict, List

import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import BufferType
from megatron.core.utils import is_float8tensor, is_torch_min_version, log_on_each_pipeline_stage
from megatron.training import get_args


logger = logging.getLogger(__name__)


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