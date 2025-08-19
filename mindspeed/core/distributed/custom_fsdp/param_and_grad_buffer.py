# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
import torch
from torch.distributed import _coalescing_manager

from megatron.core.distributed.custom_fsdp.param_and_grad_buffer import BucketStatus


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


def mark_bucket_ready(self, bucket_id: int, async_rs: bool = False) -> bool:
    """Mark the bucket ready for reduce-scatter/all-reduce, if all bucket in
    the bucket group are ready, then do the reduce-scatter/all-reduce.
    Args:
        bucket_id (int): The bucket to be marked.
        async_rs (bool, optional): Whether to do the reduce-scatter/all-reduce
            asynchronously. Defaults to False.
    Returns:
        bool: True if the bucket is go for reduce-scatter/all-reduce.
    """
    # Prepare the bucket group for gradient reduce. Note that the
    # some bucket parameters do not require grad, so we need to
    # remove them from the bucket group.
    bucket_group = self.buffer.bucket_group_of_bucket[bucket_id]
    bucket_group = [i for i in bucket_group if self.buffer.parameter_groups[i].main_grad_buffer]
    # If any bucket in the bucket group is not ready, skip the gradient reduce
    # waiting for the bucket group to be all ready before executing.
    for bucket_id in bucket_group:
        param_group = self.buffer.parameter_groups[bucket_id]
        if len(self.bucket_grad_ready_params[bucket_id]) != len(param_group.params):
            return False

    current_stream = torch.cuda.current_stream()
    reduce_scatter_stream = (
        self.cuda_stream if self.cuda_stream is not None else torch.cuda.current_stream()
    )
    reduce_scatter_stream.wait_stream(current_stream)

    dp_group = self.buffer.parameter_groups[bucket_id].main_grad_buffer.data_parallel_group
    with torch.cuda.stream(reduce_scatter_stream):
        with _coalescing_manager(dp_group, async_ops=async_rs) as coalescing_event:
            grad_shards = {}
            for bucket_id in bucket_group:
                gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                bucket = gbuf.fetch_bucket()
                scaling_factor = gbuf.gradient_scaling_factor
                reduce_op = gradient_reduce_preprocessing(
                    gbuf.data, scaling_factor, gbuf.ddp_config
                )
                bucket.data.mul_(scaling_factor)
                if gbuf.ddp_config.data_parallel_sharding_strategy == 'no_shard':
                    torch.distributed.all_reduce(
                        bucket.data, op=reduce_op, group=gbuf.data_parallel_group
                    )
                else:
                    grad_shard = gbuf.get_shard_from_bucket(bucket)
                    grad_shard = torch.empty_like(grad_shard)
                    torch.distributed.reduce_scatter_tensor(
                        output=grad_shard,
                        input=bucket.data,
                        op=reduce_op,
                        group=gbuf.data_parallel_group,
                    )
                    grad_shards[bucket_id] = grad_shard
                self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING
        coalescing_event.wait()
        for bucket_id in bucket_group:
            # Local gradient accumulate
            gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
            if gbuf.ddp_config.data_parallel_sharding_strategy != 'no_shard':
                # Gradient accumulate on local buffer
                local_buffer = gbuf.get_shard_from_local_buffer()
                local_buffer += grad_shards[bucket_id]
        reduce_scatter_view_out_event = reduce_scatter_stream.record_event()

    free_up_grad_bucket_func = {}
    for bucket_id in bucket_group:

        def get_closure(bucket_id):
            def free_up_grad_bucket():
                self.bucket_grad_ready_params[bucket_id] = set()
                gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                if gbuf.is_data_distributed:
                    gbuf.free_bucket_storage()
                self.bucket_status[bucket_id] = BucketStatus.EMPTY

            return free_up_grad_bucket

        free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

    if async_rs:
        for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
            self.grad_reduce_queue.append(
                (reduce_scatter_view_out_event, free_up_grad_bucket, bucket_id)
            )
        return True

    reduce_scatter_view_out_event.wait()
    for free_up_grad_bucket in free_up_grad_bucket_func.values():
        free_up_grad_bucket()
    return True
