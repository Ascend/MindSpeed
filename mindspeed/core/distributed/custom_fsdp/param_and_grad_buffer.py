# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import List

import torch
from torch.distributed import _coalescing_manager

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import BucketStatus


@torch.no_grad()
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
    """Scale once before SUM; HCCL has no NCCL premultiplied SUM operation."""
    if scaling_factor is None:
        return torch.distributed.ReduceOp.SUM
    if ddp_config.average_in_collective:
        return torch.distributed.ReduceOp.AVG
    grad_data.mul_(scaling_factor)
    return torch.distributed.ReduceOp.SUM


def bucket_group_gradient_reduce(
    self,
    bucket_group: List[int],
    async_op: bool = False,
    outer_fsdp_group_grad_reduce: bool = False,
) -> bool:
    """Reduce a ready bucket group using separate HCCL output buffers.

    Based on MCore 0.18 GradReducePipeline; preserve its stream, mixed
    precision, double-buffer and hybrid-sharding lifecycle.
    Args:
        bucket_group (List[int]): The bucket group to be reduced.
        async_op (bool, optional): Whether to do the reduce-scatter/all-reduce
            asynchronously. Defaults to False.
        outer_fsdp_group_grad_reduce (bool): Also reduce across outer DP groups.
    Returns:
        bool: True if the bucket is go for reduce-scatter/all-reduce.
    """
    # When using FSDP double buffer, waiting for the necessary bucket to be
    # released ensures that our double buffer will not explode due to too
    # many empty bucket requests.
    ddp_config = self.buffer.ddp_config
    mp_policy = self.buffer.mp_policy
    if ddp_config.fsdp_double_buffer:
        self._enforce_double_buffer_limit(bucket_group)

    current_stream = torch.cuda.current_stream()
    reduce_scatter_stream = self.rs_stream if self.rs_stream is not None else torch.cuda.current_stream()
    reduce_scatter_stream.wait_stream(current_stream)

    # DP-Shard Gradient Reduction
    dp_group = self.get_fsdp_buffer(bucket_group[0]).data_parallel_group
    with torch.cuda.stream(reduce_scatter_stream):
        with _coalescing_manager(dp_group):
            # List of gradient accumulation closure tasks.
            # (grad_buffer, reduced_grad)
            grad_accum_closure = []
            for bucket_id in bucket_group:
                # Get the DP-Shard gradient buffer associated with this bucket ID.
                gbuf = self.get_fsdp_buffer(bucket_id)

                # Get the unreduced gradients associated with the gradient buffer.
                unreduced_grad_bucket = gbuf.fetch_bucket(
                    dtype=mp_policy.grad_comm_dtype if gbuf.is_data_distributed else None
                )
                # NOTE(@cspades): `no_shard` or `optim`
                # Un-sharded gradient buffers accumulate un-reduced gradients locally
                # without allocating an un-sharded buffer. For custom communication
                # data-type(s), an extra un-sharded buffer needs to be allocated!
                custom_grad_comm_dtype = (
                    mp_policy.grad_comm_dtype is not None
                    and unreduced_grad_bucket.data.dtype != mp_policy.grad_comm_dtype
                )
                if not gbuf.is_data_distributed and custom_grad_comm_dtype:
                    # Create a custom communication buffer with gbuf.
                    # Introduces copy and memory overhead.
                    unreduced_grad_bucket = gbuf.allocate_bucket_storage(
                        dtype=mp_policy.grad_comm_dtype,
                        device=unreduced_grad_bucket.data.device,
                        init_values=unreduced_grad_bucket.data,
                    )
                unreduced_grad = unreduced_grad_bucket.data

                # Pre-scale unsharded bucket gradient and prepare the ReduceOp.
                scaling_factor = gbuf.gradient_scaling_factor
                reduce_op = gradient_reduce_preprocessing(unreduced_grad, scaling_factor, ddp_config)

                # Reduce-scatter or all-reduce the unsharded gradient.
                if ddp_config.data_parallel_sharding_strategy == "no_shard":
                    # All-reduce un-sharded gradients from every rank.
                    torch.distributed.all_reduce(unreduced_grad, op=reduce_op, group=gbuf.data_parallel_group)
                    if custom_grad_comm_dtype:
                        # Reduction used a temporary communication buffer.
                        grad_accum_closure.append(
                            # Un-sharded buffer data.
                            (gbuf.data, unreduced_grad)
                        )
                else:
                    # Slice a gradient shard from the communication bucket.
                    # HCCL requires reduce-scatter output to have separate storage.
                    grad_shard = torch.empty_like(gbuf.get_shard_from_bucket(unreduced_grad_bucket))

                    # Execute the reduce-scatter collective.
                    torch.distributed.reduce_scatter_tensor(
                        output=grad_shard,
                        input=unreduced_grad,
                        op=reduce_op,
                        group=gbuf.data_parallel_group,
                    )

                    # Always install the separate output, including the unsharded
                    # `optim` buffer, whose reduced slice used to alias the input.
                    grad_accum_closure.append((gbuf.get_shard_from_local_buffer(), grad_shard))

                # Mark bucket ID as CUDA work-in-progress.
                self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING

        for local_grad, reduced_grad in grad_accum_closure:
            if ddp_config.data_parallel_sharding_strategy in ["no_shard", "optim"]:
                # Copy the reduced gradient into the main gradient buffer.
                local_grad.copy_(reduced_grad)
            else:
                # Accumulate the reduced gradient into the local gradient buffer.
                # Accumulation data-type is type-promoted with respect to the
                # accumulated gradient and the buffer main_grads_dtype.
                local_grad += reduced_grad

        # Record a checkpoint for the event to synchronize against the reduce-scatter stream.
        reduce_scatter_view_out_event = reduce_scatter_stream.record_event()

    # DP-Outer Gradient Reduction
    if outer_fsdp_group_grad_reduce:
        # Wait on the DP-Shard reduction before further reduction.
        self.outer_fsdp_group_grad_reduce_stream.wait_stream(reduce_scatter_stream)
        outer_fsdp_group = self.buffer.dist_index.get_outer_fsdp_group()
        with torch.cuda.stream(self.outer_fsdp_group_grad_reduce_stream):
            with _coalescing_manager(outer_fsdp_group):
                # List of gradient accumulation closure tasks.
                # (grad_buffer, reduced_grad)
                grad_accum_closure = []
                for bucket_id in bucket_group:
                    # Skip gradient scaling for DP-Outer, because the
                    # (DP-Shard, DP-Outer) scaling is already applied.
                    if ddp_config.average_in_collective:
                        reduce_op = torch.distributed.ReduceOp.AVG
                    else:
                        reduce_op = torch.distributed.ReduceOp.SUM

                    # (DP-Shard, DP-Outer) if HFSDP, otherwise just DP-Shard for HSDP
                    main_grad_buffer = self.buffer.parameter_groups[bucket_id].main_grad_buffer

                    # FSDP buffer can be un-sharded or sharded for HSDP, but sharded for HFSDP.
                    # TODO(@cspades): For `optim`, we don't need to reduce the local un-sharded
                    # gradient, just the shard updated via reduce-scatter.
                    fsdp_grad_buffer = self.get_fsdp_buffer(bucket_id)
                    unreduced_grad = fsdp_grad_buffer.data
                    assert main_grad_buffer.dtype == fsdp_grad_buffer.dtype, (
                        "Main and DP-Shard gradient buffer must share the exact same dtype."
                    )

                    # Cast DP-Shard gradient to communication dtype if specified and necessary.
                    custom_grad_comm_dtype = (
                        mp_policy.grad_comm_dtype is not None and unreduced_grad.dtype != mp_policy.grad_comm_dtype
                    )
                    if custom_grad_comm_dtype:
                        # Allocate a custom communication buffer with the HSDP gradient
                        # communication buffer. Introduces copy and memory overhead.
                        hsdp_comm_gbuf = self.buffer.parameter_groups[bucket_id].hsdp_comm_gbuf
                        unreduced_grad = hsdp_comm_gbuf.allocate_bucket_storage(
                            # Allocate memory for the sharded or un-sharded
                            # gradient reduced over DP-Shard.
                            shard=fsdp_grad_buffer.is_data_distributed,
                            dtype=mp_policy.grad_comm_dtype,
                            device=unreduced_grad.device,
                            init_values=unreduced_grad,
                        ).data

                    # All-reduce or reduce-scatter the DP-Shard gradients across DP-Outer.
                    if ddp_config.outer_dp_sharding_strategy != "no_shard":
                        # Retrieve the (DP-Outer, DP-Shard) gradient shard from the
                        # main gradient buffer which shards across the entire DP group,
                        # i.e. across all DP-Shard and DP-Outer ranks.
                        main_grad_shard = main_grad_buffer.get_shard_from_local_buffer()
                        # HFSDP also needs a disjoint output for HCCL.
                        output_buffer = torch.empty_like(main_grad_shard, dtype=unreduced_grad.dtype)
                        # Reduce-scatter the FSDP gradient buffer shard further
                        # into the (DP-Outer, DP-Shard) gradient shard.
                        torch.distributed.reduce_scatter_tensor(
                            output=output_buffer,
                            input=unreduced_grad,
                            op=reduce_op,
                            group=outer_fsdp_group,
                        )
                        grad_accum_closure.append((main_grad_shard, output_buffer))
                    else:  # HSDP -> main_grad_buffer = (DP-Shard,)
                        # No DP-Outer sharding, so all-reduce FSDP gradients across DP-Outer.
                        # All FSDP buffers will have reduced un-sharded or sharded gradients.
                        torch.distributed.all_reduce(unreduced_grad, group=outer_fsdp_group, op=reduce_op)
                        if custom_grad_comm_dtype:
                            # Reduction used a temporary communication buffer.
                            grad_accum_closure.append((main_grad_buffer.data, unreduced_grad))

            for main_grad_buffer, reduced_grad in grad_accum_closure:
                # Update the (DP-Outer, DP-Shard) gradient shard in the main gradient buffer.
                # No accumulation should happen in the (DP-Shard, DP-Outer) gradient buffer.
                main_grad_buffer.copy_(reduced_grad)

        reduce_scatter_view_out_event = self.outer_fsdp_group_grad_reduce_stream.record_event()

    free_up_grad_bucket_func = {}
    for bucket_id in bucket_group:

        def get_closure(bucket_id):
            def free_up_grad_bucket():
                # Empty the set of parameters that are ready for gradient reduction.
                self.bucket_grad_ready_params[bucket_id] = set()
                gbuf = self.get_fsdp_buffer(bucket_id)
                # Free the memory backing the temporarily-allocated communication
                # bucket associated with this buffer. Only exists for sharded
                # gradient buffers, or if a custom gradient data-type is used!
                gbuf.free_bucket_storage()
                # Gradient reduction completed, can de-reference param.main_grad.
                gbuf.reset_param_main_grad()
                hsdp_comm_gbuf = self.buffer.parameter_groups[bucket_id].hsdp_comm_gbuf
                if hsdp_comm_gbuf is not None:
                    # Also de-allocate any communication buffers used for H(F)SDP.
                    hsdp_comm_gbuf.free_bucket_storage()
                # Mark the bucket as deallocated / empty.
                self.bucket_status[bucket_id] = BucketStatus.EMPTY

            return free_up_grad_bucket

        free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

    if async_op:
        for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
            self.grad_reduce_queue.append((reduce_scatter_view_out_event, free_up_grad_bucket, bucket_id))
        return True

    reduce_scatter_view_out_event.wait()
    for free_up_grad_bucket in free_up_grad_bucket_func.values():
        free_up_grad_bucket()
    return True
