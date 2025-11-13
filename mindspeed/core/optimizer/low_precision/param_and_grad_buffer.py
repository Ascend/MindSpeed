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
        dp_world_size = 1
        if hasattr(self, 'data_parallel_group') and self.data_parallel_group is not None:
            try:
                dp_world_size = torch.distributed.get_world_size(self.data_parallel_group)
            except RuntimeError:
                dp_world_size = 1
        if (
            not quant_grads_enabled
            or dp_world_size <= 1
        ):
            return start_grad_sync(self)

        original_check_nan = self.ddp_config.check_for_nan_in_grad
        original_check_large = self.ddp_config.check_for_large_grads
        self.ddp_config.check_for_nan_in_grad = False
        self.ddp_config.check_for_large_grads = False

        try:
            assert (
                self.grad_reduce_handle is None
            ), 'Should not have multiple communication calls outstanding at once'

            communication_group = self.data_parallel_group
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
        finally:
            self.ddp_config.check_for_nan_in_grad = original_check_nan
            self.ddp_config.check_for_large_grads = original_check_large

    return quant_start_grad_sync


def quant_grad_finish_grad_sync_wrapper(finish_grad_sync):
    @wraps(finish_grad_sync)
    def quant_finish_grad_sync(self):
        quant_args = get_full_args()
        quant_grads_enabled = getattr(quant_args, 'quant_grads', False)
        dp_world_size = 1
        if hasattr(self, 'data_parallel_group') and self.data_parallel_group is not None:
            try:
                dp_world_size = torch.distributed.get_world_size(self.data_parallel_group)
            except RuntimeError:
                dp_world_size = 1
        if (
            not quant_grads_enabled
            or self.ddp_config.use_distributed_optimizer
            or self.ddp_config.num_distributed_optimizer_instances > 1
            or dp_world_size <= 1
        ):
            return finish_grad_sync(self)
        return finish_grad_sync(self)

    return quant_finish_grad_sync


def param_and_grad_buffer_init_wrapper(init_func):
    @wraps(init_func)
    def param_and_grad_buffer_init_func(
            self,
            ddp_config: DistributedDataParallelConfig,
            param_dtype: torch.dtype,
            grad_dtype: torch.dtype,
            params: List[torch.nn.Parameter],
            data_parallel_group: torch.distributed.ProcessGroup,
            bucket_size: int,
            param_to_name: Dict[torch.nn.Parameter, str],
            gradient_scaling_factor: float,
            param_indices: List[int], ):
        init_func(self, ddp_config, param_dtype, grad_dtype, params, data_parallel_group, bucket_size, param_to_name,
                  gradient_scaling_factor, param_indices)
        args = get_args()
        print(f'param_and_grad_buffer_init_wrapper' + 'S0'*300)
        if args.quant_grads:
            quant_dtype_token = getattr(args, 'quant_grads_dtype', 'fp16')
            if isinstance(quant_dtype_token, str):
                quant_dtype_token = quant_dtype_token.lower()
            qtype_token = 'bf16' if quant_dtype_token == 'bf16' else 'fp16'
            print(f'param_and_grad_buffer_init_wrapper' + 'S1'*300)
            bucket_params = set()
            cur_bucket_id = 0
            self.scaling_grads = [[] for _ in range(len(self.buckets))]
            scales = torch.ones((len(params),), device='cuda')
            scale_start_index = 0
            scale_end_index = 0
            for idx, param in enumerate(params[::-1]):
                if not param.requires_grad:
                    continue
                data_start_index, data_end_index, bucket_id = self.param_index_map[param]

                param.main_grad = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.GRAD
                )
                print(f'param_and_grad_buffer_init_wrapper' + 'S2'*300)
                if bucket_id != cur_bucket_id:
                    print(f'param_and_grad_buffer_init_wrapper' + 'S3'*300)
                    setattr(self.buckets[cur_bucket_id], "scaling_grads", self.scaling_grads[cur_bucket_id])
                    setattr(self.buckets[cur_bucket_id], "scales", scales[scale_start_index:scale_end_index])

                    bucket_params = set()
                    assert bucket_id == cur_bucket_id + 1
                    cur_bucket_id = bucket_id
                    scale_start_index = scale_end_index

                scale_meta = ScaleMeta(qtype_token, param.main_grad, param.main_grad.numel())
                param.main_grad.meta = scale_meta
                param.main_grad.meta.scale = scales[idx:idx + 1]
                param.main_grad.meta.scale_inv = torch.ones_like(param.main_grad.meta.scale)

                self.scaling_grads[cur_bucket_id].append(param.main_grad)
                scale_end_index += 1

                bucket_params.add(param)
            print(f'param_and_grad_buffer_init_wrapper' + 'S4'*300)
            if len(bucket_params) > 0:
                print(f'param_and_grad_buffer_init_wrapper' + 'S5'*300)
                setattr(self.buckets[cur_bucket_id], "scaling_grads", self.scaling_grads[cur_bucket_id])
                setattr(self.buckets[cur_bucket_id], "scales", scales[scale_start_index:scale_end_index])

    return param_and_grad_buffer_init_func

def bucket_init_wrapper(init_func):
    @wraps(init_func)
    def bucket_init_func(self, *args, **kwargs):
        init_func(self, *args, **kwargs)
        args = get_args()
        if args.quant_grads:
            self.scaling_grads = None
            self.scales = None

    return bucket_init_func


        
def bucket_start_grad_sync(self):
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


    args = get_args()
    if args.quant_grads and len(self.scaling_grads) > 0:
        old_grad_scales = self.scales.clone()
        torch.distributed.all_reduce(
            self.scales,
            op=torch.distributed.ReduceOp.MIN,
            group=self.data_parallel_group,
            async_op=self.ddp_config.overlap_grad_reduce
        )

        need_requant = (old_grad_scales != self.scales).tolist()
        for need, g, new_scale, old_scale in zip(need_requant, self.scaling_grads, self.scales, old_grad_scales):
            if need:
                g.meta.scale.copy_(new_scale)
                g.meta.scale_inv.copy_(1 / g.meta.scale)
                g.data.copy_((g.data.float() / old_scale * g.meta.scale))
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