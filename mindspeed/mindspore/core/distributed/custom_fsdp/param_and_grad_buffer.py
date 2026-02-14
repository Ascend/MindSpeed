# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import traceback
from enum import Enum
from typing import Any, List, Optional

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import _p_assert, _free_storage, PrefetchOrder, BucketStatus, gradient_reduce_preprocessing


def all_gather_params_wo_coalescing(
    self,
    params: List[torch.Tensor],
    prefetch: bool = False,
    prefetch_order: PrefetchOrder = PrefetchOrder.FORWARD_PASS_ORDER,
    suggested_AG_prefetch_size: Optional[int] = None,
    ):
    if len(params) == 0:
        return

    ag_buckets = [self.buffer.param_to_param_group[item] for item in params]
    ag_buckets = list(sorted(set(ag_buckets)))
    parameter_groups = self.buffer.parameter_groups

    if prefetch:

        def next_bucket_id(ag_buckets):
            if prefetch_order == PrefetchOrder.FORWARD_PASS_ORDER:
                bucket_id = ag_buckets[0] + 1
                for i in ag_buckets[1:]:
                    if i != bucket_id:
                        break
                    bucket_id += 1
            else:
                bucket_id = ag_buckets[-1] - 1
                for i in reversed(ag_buckets[:-1]):
                    if i != bucket_id:
                        break
                    bucket_id -= 1
            if bucket_id < 0 or bucket_id >= self.buffer.num_buckets:
                return None
            return bucket_id

        if suggested_AG_prefetch_size is not None:
            bucket_id = next_bucket_id(ag_buckets)
            while bucket_id is not None:
                all_gather_size = sum(
                    [
                        parameter_groups[i].model_weight_buffer.bucket_index.size
                        for i in ag_buckets
                    ]
                )
                if all_gather_size >= suggested_AG_prefetch_size:
                    break
                ag_buckets.extend(self.buffer.bucket_group_of_bucket[bucket_id])
                ag_buckets = list(sorted(set(ag_buckets)))
                bucket_id = next_bucket_id(ag_buckets)
        else:
            bucket_id = next_bucket_id(ag_buckets)
            if bucket_id is not None:
                ag_buckets.extend(self.buffer.bucket_group_of_bucket[bucket_id])
                ag_buckets = list(sorted(set(ag_buckets)))

    ag_buckets = [i for i in ag_buckets if self.bucket_status[i] == BucketStatus.EMPTY]
    if len(ag_buckets) == 0:
        return

    bucket_group_to_buckets = {}
    for bucket_id in ag_buckets:
        group_id = self.bucket_to_bucket_group[bucket_id]
        if group_id not in bucket_group_to_buckets:
            bucket_group_to_buckets[group_id] = []
        bucket_group_to_buckets[group_id].append(bucket_id)

    # Coalesced communication not supported
    # Using non-coalesced operations instead 
    for _, buckets in bucket_group_to_buckets.items():
        param_group = parameter_groups[buckets[0]]
        dp_group = param_group.model_weight_buffer.data_parallel_group
        for bucket_id in buckets:
            self.all_gather_bucket_and_set_items(bucket_id, async_op=True)


def mark_bucket_ready_wo_coalescing(self, bucket_id: int, async_rs: bool = False) -> bool:
    bucket_group = self.buffer.bucket_group_of_bucket[bucket_id]
    bucket_group = [i for i in bucket_group if self.buffer.parameter_groups[i].main_grad_buffer]
    for bucket_id in bucket_group:
        param_group = self.buffer.parameter_groups[bucket_id]
        if len(self.bucket_grad_ready_params[bucket_id]) != len(param_group.params):
            return False

    grad_shards = {}
    reduce_scatter_view_out_events = {}

    # Coalesced communication not supported
    # Using non-coalesced operations instead 
    for bucket_id in bucket_group:
        gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
        bucket = gbuf.fetch_bucket()
        scaling_factor = gbuf.gradient_scaling_factor
        reduce_op = gradient_reduce_preprocessing(
            gbuf.data, scaling_factor, gbuf.ddp_config
        )
        bucket.data.mul_(scaling_factor)
        if gbuf.ddp_config.data_parallel_sharding_strategy == 'no_shard':
            reduce_event = torch.distributed.all_reduce(
                bucket.data, op=reduce_op, group=gbuf.data_parallel_group, async_op=async_rs,
            )
        else:
            grad_shard = gbuf.get_shard_from_bucket(bucket)

            grad_shard = torch.empty_like(grad_shard)
            reduce_event = torch.distributed.reduce_scatter_tensor(
                output=grad_shard,
                input=bucket.data,
                op=reduce_op,
                group=gbuf.data_parallel_group,
                async_op=async_rs,
            )
            grad_shards[bucket_id] = grad_shard
        if reduce_event:
            reduce_scatter_view_out_events[bucket_id] = reduce_event
        self.bucket_status[bucket_id] = BucketStatus.COMMUNICATING

    free_up_grad_bucket_func = {}
    for bucket_id in bucket_group:

        def get_closure(bucket_id):
            def free_up_grad_bucket():
                self.bucket_grad_ready_params[bucket_id] = set()
                gbuf = self.buffer.parameter_groups[bucket_id].main_grad_buffer
                if gbuf.ddp_config.data_parallel_sharding_strategy != 'no_shard':
                    local_buffer = gbuf.get_shard_from_local_buffer()
                    local_buffer += grad_shards[bucket_id]
                    del grad_shards[bucket_id]
                if gbuf.is_data_distributed:
                    gbuf.free_bucket_storage()
                self.bucket_status[bucket_id] = BucketStatus.EMPTY

            return free_up_grad_bucket

        free_up_grad_bucket_func[bucket_id] = get_closure(bucket_id)

    if async_rs:
        for bucket_id, free_up_grad_bucket in free_up_grad_bucket_func.items():
            self.grad_reduce_queue.append(
                (reduce_scatter_view_out_events[bucket_id], free_up_grad_bucket, bucket_id)
            )
        return True

    for _, event in reduce_scatter_view_out_events.items():
        event.wait()
    for free_up_grad_bucket in free_up_grad_bucket_func.values():
        free_up_grad_bucket()
    return True


def zero_grad(self):
    for _, param in self.optimizer_named_parameters:
        # MindSpore Tensors do not have a '_base' attribute
        # Using '_grad_base' as a custom flag
        if param.grad is not None and not getattr(param, '_grad_base', False):
            _free_storage(param.grad)
        param.grad = None
        continue

    for group in self.parameter_groups:
        if group.main_grad_buffer is None:
            continue
        group.main_grad_buffer.data.zero_()


def update_main_grads(self):
    for _, param in self.optimizer_named_parameters:
        param.reset_attribute()
        orig_param = param.orig_param
        group = self.parameter_groups[self.param_to_param_group[orig_param]]
        item_id = group.main_grad_buffer.param_idx[orig_param]
        optimizer_grad = group.main_grad_buffer.get_item(
            item_id, only_shard=group.main_weight_buffer.is_data_distributed
        )
        setattr(
            param,
            'grad',
            optimizer_grad.to(param.dtype) if optimizer_grad.numel() > 0 else None,
        )
        if param.grad is not None:
            # MindSpore Tensors do not have a '_base' attribute
            # Using '_grad_base' as a custom flag
            setattr(param, '_grad_base', True)