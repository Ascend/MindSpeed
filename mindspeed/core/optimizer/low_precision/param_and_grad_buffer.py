import os
from enum import Enum
from functools import wraps
from typing import Dict, List
from contextlib import nullcontext
import torch
from megatron.training import get_args
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import (
    BufferType,
    dist_all_gather_func,
    dist_reduce_scatter_func,
    shard_buffer,
)
from mindspeed.core.optimizer.low_precision.quant_adamw import ScaleMeta
from mindspeed.args_utils import get_full_args


def quant_grad_param_and_grad_buffer_init_wrapper(init_func):
    @wraps(init_func)
    def quant_grad_param_and_grad_buffer_init(self, ddp_config, param_dtype, grad_dtype, *args, **kwargs):
        quant_args = get_full_args()
        quant_grads_enabled = getattr(quant_args, 'quant_grads', False)
        qdtype = None
        if quant_grads_enabled:
            qdtype = getattr(quant_args, 'quant_grads_dtype', None)
            if isinstance(qdtype, str):
                qdtype = qdtype.lower()
            grad_dtype = torch.bfloat16 if qdtype == 'bf16' else torch.float16

        init_func(self, ddp_config, param_dtype, grad_dtype, *args, **kwargs)

        if not quant_grads_enabled:
            return

        # Default NaN/Inf checks use the unquantized bucket values; disable them and rely on
        # higher-level AMP/non-finite handling when quant grads are enabled.
        self.ddp_config.check_for_nan_in_grad = False
        self.ddp_config.check_for_large_grads = False

        # Gradients in each bucket need dedicated scale tensors so we can keep quantization
        # metadata in sync across the data-parallel group.
        device = self.grad_data.device
        scale_token = 'bf16' if qdtype == 'bf16' else 'fp16'

        bucket_grad_lists = [[] for _ in range(len(self.buckets))]
        for param in getattr(self, 'params', [])[::-1]:
            if not getattr(param, 'requires_grad', False):
                continue
            _, _, bucket_id = self.param_index_map[param]
            bucket_grad_lists[bucket_id].append((param, param.main_grad))

        for bucket_id, grad_list in enumerate(bucket_grad_lists):
            bucket = self.buckets[bucket_id]
            # Initialise scaling structures even when the bucket is empty so downstream logic
            # has consistent attributes to check.
            bucket.scaling_grads = []
            if not grad_list:
                bucket.scales = torch.empty(0, device=device)
                continue

            bucket.scales = torch.ones(len(grad_list), device=device, dtype=torch.float32)
            for idx, (param, grad_tensor) in enumerate(grad_list):
                # Attach per-gradient quantization metadata.
                scale_meta = ScaleMeta(scale_token, grad_tensor, grad_tensor.numel())
                grad_tensor.meta = scale_meta
                scale_slice = bucket.scales[idx:idx + 1]
                scale_meta.scale = scale_slice
                scale_inv = torch.ones_like(scale_slice)
                scale_inv.copy_(1 / scale_slice)
                scale_meta.scale_inv = scale_inv
                bucket.scaling_grads.append(grad_tensor)
                # Ensure downstream hooks can always locate the quantized view on the parameter.
                setattr(param, "quant_grad", grad_tensor)

    return quant_grad_param_and_grad_buffer_init


def quant_grad_start_grad_sync_wrapper(start_grad_sync):
    @wraps(start_grad_sync)
    def quant_start_grad_sync(self):
        quant_args = get_full_args()
        quant_grads_enabled = getattr(quant_args, 'quant_grads', False)

        if not quant_grads_enabled:
            return start_grad_sync(self)

        assert (
            self.grad_reduce_handle is None
        ), 'Should not have multiple communication calls outstanding at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        if self.ddp_config.check_for_nan_in_grad:
            global_rank = torch.distributed.get_rank()
            for bucket in self.buckets:
                norm = bucket.grad_data.norm(p=2)
                assert not norm.isnan(), (
                    f'Rank {global_rank}: found NaN in local grad norm in '
                    f'backward pass before data-parallel communication collective. '
                    f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
                )
        # Match the same communication group that the underlying bucket group
        # will use for gradient reduction, falling back safely when running
        # with distributed optimizer which may not populate data_parallel_group.
        communication_group = getattr(self, "data_parallel_group", None)
        if getattr(self.ddp_config, "use_distributed_optimizer", False):
            communication_group = getattr(
                self, "intra_distributed_optimizer_instance_group", communication_group
            )
        if communication_group is None and self.buckets:
            communication_group = getattr(self.buckets[0], "data_parallel_group", None)
        if communication_group is None:
            raise AttributeError(
                "No communication group found for quantized grad scale synchronization"
            )
        for bucket in self.buckets:
            scaling_grads = getattr(bucket, 'scaling_grads', None)
            if not scaling_grads:
                continue

            old_scales = bucket.scales.clone()
            torch.distributed.all_reduce(
                bucket.scales,
                op=torch.distributed.ReduceOp.MIN,
                group=communication_group,
                async_op=False,
            )
            need_requant = torch.ne(old_scales, bucket.scales)
            if not torch.any(need_requant).item():
                continue

            for idx, grad_tensor in enumerate(scaling_grads):
                if idx >= need_requant.numel() or not need_requant[idx].item():
                    continue

                grad_meta = getattr(grad_tensor, 'meta', None)
                if grad_meta is None:
                    continue

                new_scale = bucket.scales[idx:idx + 1]
                old_scale = old_scales[idx:idx + 1]

                grad_meta.scale.copy_(new_scale)
                if getattr(grad_meta, 'scale_inv', None) is None or grad_meta.scale_inv.shape != grad_meta.scale.shape:
                    grad_meta.scale_inv = torch.ones_like(grad_meta.scale)

                if torch.all(old_scale != 0).item():
                    updated = grad_tensor.data.float()
                    ratio = (grad_meta.scale / old_scale).to(updated.dtype)
                    updated.mul_(ratio)
                    grad_tensor.data.copy_(updated.to(dtype=grad_tensor.dtype))
                else:
                    grad_tensor.data.zero_()

                safe_scale = grad_meta.scale.clone()
                scale_inv = torch.zeros_like(safe_scale)
                non_zero_mask = safe_scale != 0
                scale_inv[non_zero_mask] = (1.0 / safe_scale[non_zero_mask])
                grad_meta.scale_inv.copy_(scale_inv)

        # Delegate the actual gradient communication to the original implementation so
        # overlap and distributed-optimizer semantics remain unchanged.
        return start_grad_sync(self)

    return quant_start_grad_sync
