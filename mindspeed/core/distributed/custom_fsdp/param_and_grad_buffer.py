# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
from typing import List
import torch
from torch.distributed import _coalescing_manager
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import BucketStatus


@torch.no_grad()
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
    """
    Gradient reduce preprocessing for gradient averaging and gradient scaling.
    """

    if scaling_factor is None:
        reduce_op = torch.distributed.ReduceOp.SUM
    elif ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG
    elif ddp_config.gradient_reduce_div_fusion and grad_data.dtype != torch.bfloat16:
        reduce_op = torch.distributed.ReduceOp.SUM
    else:
        grad_data.mul_(scaling_factor)
        reduce_op = torch.distributed.ReduceOp.SUM

    return reduce_op


def _bucket_group_gradient_reduce(
        self,
        bucket_group: List[int],
        async_op: bool = False,
        outer_fsdp_group_grad_reduce: bool = False,
    ) -> bool:
    """Mark the bucket ready for reduce-scatter/all-reduce, if all bucket in
    the bucket group are ready, then do the reduce-scatter/all-reduce.
    Args:
        bucket_id (int): The bucket to be marked.
        bucket_group (List[int]): The bucket group to be reduced.
        async_op (bool, optional): Whether to do the reduce-scatter/all-reduce
            asynchronously. Defaults to False.
    Returns:
        bool: True if the bucket is go for reduce-scatter/all-reduce.
    """
    # When using FSDP double buffer, waiting for the necessary bucket to be
    # released ensures that our double buffer will not explode due to too
    # many empty bucket requests.
    ddp_config = self.buffer.ddp_config
    if ddp_config.fsdp_double_buffer:
        self._enforce_double_buffer_limit(bucket_group)

    current_stream = torch.cuda.current_stream()
    reduce_scatter_stream = (
        self.rs_stream if self.rs_stream is not None else torch.cuda.current_stream()
    )
    reduce_scatter_stream.wait_stream(current_stream)

    dp_group = self.get_fsdp_buffer(bucket_group[0]).data_parallel_group
    with torch.cuda.stream(reduce_scatter_stream):
        with _coalescing_manager(dp_group):
            grad_buffer = []
            reduced_grad = []
            for bucket_id in bucket_group:
                # Fetch pre-allocated main gradient bucket.
                gbuf = self.get_fsdp_buffer(bucket_id)
                bucket = gbuf.fetch_bucket()
                # Scale gradients.
                scaling_factor = gbuf.gradient_scaling_factor
                reduce_op = gradient_reduce_preprocessing(
                    gbuf.data, scaling_factor, gbuf.ddp_config
                )
                bucket.data.mul_(scaling_factor)
                if not gbuf.is_data_distributed:
                    # All-reduce the gradients on every rank. No scattering
                    # or sharding necessary.
                    torch.distributed.all_reduce(
                        bucket.data, op=reduce_op, group=gbuf.data_parallel_group
                    )
                else:
                    # Get the shard of the gradient from the pre-allocated bucket.
                    # The reduced gradient will be scattered into this shard of the
                    # bucket managed by the sharded buffer on this rank.
                    grad_shard = gbuf.get_shard_from_bucket(bucket)
                    # pylint: disable=C0301
                    # The `grad_shard`` is part of `bucket.data`` and the following
                    # new empty is important for memory safety, when using
                    # TORCH_NCCL_AVOID_RECORD_STREAMS=1.
                    # For reference: https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486
                    if not self.buffer.ddp_config.fsdp_double_buffer:
                        grad_shard = torch.empty_like(grad_shard)
                    # Reduce-scatter gradients on the FSDP group.
                    torch.distributed.reduce_scatter_tensor(
                        output=grad_shard,
                        input=bucket.data,
                        op=reduce_op,
                        group=gbuf.data_parallel_group,
                    )
                    reduced_grad.append(grad_shard)
                    grad_buffer.append(gbuf.get_shard_from_local_buffer())
                self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING
        for local_grad, reduced_grad in zip(grad_buffer, reduced_grad):
            # Accumulate the reduced gradient shard into the local gradient buffer,
            # when ZeRO-2 (gradient sharding) is enabled. Otherwise, bucket.data
            # is equivalent to the buffer data and will have been all-reduced.
            local_grad += reduced_grad
        # Record a checkpoint for the event to synchronize against the reduce-scatter stream.
        reduce_scatter_view_out_event = reduce_scatter_stream.record_event()

    # Outer-DP group gradient reduction.
    if outer_fsdp_group_grad_reduce:
        self.outer_fsdp_group_grad_reduce_stream.wait_stream(reduce_scatter_stream)
        outer_fsdp_group = self.buffer.dist_index.get_outer_fsdp_group()
        with torch.cuda.stream(self.outer_fsdp_group_grad_reduce_stream):
            with _coalescing_manager(outer_fsdp_group):
                reduced_grad = []
                for bucket_id in bucket_group:
                    if ddp_config.average_in_collective:
                        reduce_op = torch.distributed.ReduceOp.AVG
                    else:
                        reduce_op = torch.distributed.ReduceOp.SUM

                    # All-reduce/reduce-scatter the gradients on every rank
                    # in the outer-DP group.
                    gbuf = self.get_fsdp_buffer(bucket_id)
                    if ddp_config.outer_dp_sharding_strategy != "no_shard":
                        # Outer-DP sharding is only supported for fully-sharded inner-DP.
                        assert ddp_config.data_parallel_sharding_strategy != "no_shard"
                        # Retrieve the (DP-Outer, DP-Shard) gradient shard from the
                        # main gradient buffer which shards across the entire DP group,
                        # i.e. across all DP-Shard and DP-Outer ranks.
                        grad_full_shard = self.buffer.parameter_groups[
                            bucket_id
                        ].main_grad_buffer.get_shard_from_local_buffer()
                        # NOTE: This is a fix for convergence, needed to make
                        # sure NCCL reduce-scatter inplace didn't seem
                        # to work correctly
                        grad_full_shard = torch.empty_like(grad_full_shard)
                        reduced_grad.append(grad_full_shard)
                        # Reduce-scatter the FSDP gradient buffer shard further
                        # into the (DP-Outer, DP-Shard) gradient shard.
                        torch.distributed.reduce_scatter_tensor(
                            output=grad_full_shard,
                            input=gbuf.data,
                            op=reduce_op,
                            group=outer_fsdp_group,
                        )
                    else:
                        # No outer-DP sharding, so just all-reduce the FSDP gradient
                        # buffer shard into itself.
                        torch.distributed.all_reduce(
                            gbuf.data, group=outer_fsdp_group, op=reduce_op
                        )
            for bucket_id, grad_full_shard in zip(bucket_group, reduced_grad):
                # Update the (DP-Outer, DP-Shard) gradient shard in the main gradient buffer.
                self.buffer.parameter_groups[
                    bucket_id
                ].main_grad_buffer.get_shard_from_local_buffer().copy_(grad_full_shard)
        reduce_scatter_view_out_event = self.outer_fsdp_group_grad_reduce_stream.record_event()

    free_up_grad_bucket_func = {}
    for bucket_id in bucket_group:

        def get_closure(bucket_id):
            def free_up_grad_bucket():
                # Empty the set of parameters that are ready for gradient reduction.
                self.bucket_grad_ready_params[bucket_id] = set()
                gbuf = self.get_fsdp_buffer(bucket_id)
                if gbuf.is_data_distributed:
                    # Free the memory backing the temporarily-allocated bucket
                    # associated with this buffer.
                    gbuf.free_bucket_storage()
                # Mark the bucket as deallocated / empty.
                self.bucket_status[bucket_id] = BucketStatus.EMPTY

            return free_up_grad_bucket

        free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

    if async_op:
        for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
            self.grad_reduce_queue.append(
                (reduce_scatter_view_out_event, free_up_grad_bucket, bucket_id)
            )
        return True

    reduce_scatter_view_out_event.wait()
    for free_up_grad_bucket in free_up_grad_bucket_func.values():
        free_up_grad_bucket()
    return True