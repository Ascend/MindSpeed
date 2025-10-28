from functools import wraps
from typing import List, Optional, Union

import torch
import torch.distributed
from torch import inf

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale,
    )

    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        from megatron.core.utils import (
            local_multi_tensor_applier,
            local_multi_tensor_l2_norm,
            local_multi_tensor_scale,
        )

        multi_tensor_applier = local_multi_tensor_applier
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale

from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor


def _ensure_group(group: Optional[torch.distributed.ProcessGroup]) -> Optional[torch.distributed.ProcessGroup]:
    if group is not None:
        return group
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.group.WORLD
    return None


def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    #print('get_grad_norm_fp32'+'C0'*300)
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    data_parallel_group = None
    float_grads = []
    for grad in grads_for_norm:
        if grad is None:
            continue
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)
        float_grads.append(to_local_if_dtensor(grad))

    norm_type = float(norm_type)
    grad_stats_parallel_group = _ensure_group(grad_stats_parallel_group)

    if norm_type == inf:
        total_norm = 0.0
        if float_grads:
            total_norm = max(float(grad.abs().max()) for grad in float_grads)
        total_norm_tensor = torch.tensor([total_norm], dtype=torch.float, device='cuda')
        if data_parallel_group is not None:
            torch.distributed.all_reduce(
                total_norm_tensor, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
            )
        if grad_stats_parallel_group is not None:
            torch.distributed.all_reduce(
                total_norm_tensor, op=torch.distributed.ReduceOp.MAX, group=grad_stats_parallel_group
            )
        return total_norm_tensor.item()

    total_norm = 0.0
    if norm_type == 2.0 and float_grads:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        grad_norm, _ = multi_tensor_applier(
            l2_norm_impl,
            dummy_overflow_buf,
            [float_grads],
            False,
        )
        total_norm += float(grad_norm**norm_type)
    else:
        for grad in float_grads:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += float(grad_norm**norm_type)

    total_norm_tensor = torch.tensor([total_norm], dtype=torch.float, device='cuda')
    if data_parallel_group is not None:
        torch.distributed.all_reduce(
            total_norm_tensor, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    if grad_stats_parallel_group is not None:
        torch.distributed.all_reduce(
            total_norm_tensor, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
    return total_norm_tensor.item() ** (1.0 / norm_type)


def clip_grad_by_total_norm_fp32_wrapper(func):
    @wraps(func)
    def clip_grad_by_total_norm_fp32(
        parameters: Union[List[torch.Tensor], torch.Tensor],
        max_norm: Union[int, float],
        total_norm: float,
        use_decoupled_grad: bool = False,
    ): 
        #print('clip_grad_by_total_norm_fp32_wrapper'+'C1'*300)
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if use_decoupled_grad:
            grads = []
            for param in parameters:
                grad = getattr(param, 'decoupled_grad', None)
                if grad is not None:
                    grads.append(grad)
            if grads and total_norm > 0.0:
                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
                    grads_local = [to_local_if_dtensor(g) for g in grads]
                    multi_tensor_applier(
                        multi_tensor_scale_impl,
                        dummy_overflow_buf,
                        [grads_local, grads_local],
                        clip_coef,
                    )
        return func(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)

    return clip_grad_by_total_norm_fp32
