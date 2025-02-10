# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Iterable
import math
import amp_C
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.multi_tensor_apply import multi_tensor_applier
from mindspeed.core.distributed.layerzero.zero3._common_utils import _is_zero3_flattened


@torch.no_grad()
def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    Return the gradient norm of parameters ``param`` s, where the gradients are viewed as a single vector.

    The returned norm is in FP32 even if parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)
    grads = [param.grad for param in params_with_grad]
    grad_dtypes = {grad.dtype for grad in grads}
    if len(grad_dtypes) != 1:
        raise ValueError(
            f"Requires uniform dtype across all gradients but got {grad_dtypes}"
        )
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(
                    grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm


def clip_grad_norm(params, max_norm, norm_type=2, process_group=dist.group.WORLD):
    '''
    For distributed ZERO optimizers, the gradient norm is calculated since the parameter/gradient 
    is distributed across the individual ranks, Additional communication is required
    It is worth noting here that the grad_norm is divided by world_size approximate DDP
    #! ZeRO-managed parameters and non-ZeRO-managed parameters are handled separately
    '''
    if not max_norm > 0.:
        raise ValueError("clip_grad should be a number greater than 0.0")

    if isinstance(params, torch.Tensor):
        params = [params]
    norm_type = float(norm_type)
    device = params[0].device
    sharded_params = set(p for p in params if _is_zero3_flattened(p))
    non_sharded_params = set(p for p in params if p not in sharded_params)

    local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(device)
    local_nonsharded_norm = _get_grad_norm(
        non_sharded_params, norm_type).to(device)
    if norm_type == math.inf:
        total_norm = (
            torch.maximum(local_sharded_norm, local_nonsharded_norm)
            if local_nonsharded_norm is not None
            else local_sharded_norm
        )
        dist.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=process_group
        )
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=process_group)
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        if local_nonsharded_norm is not None:
            total_norm += local_nonsharded_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    grads = list(set(param.grad for param in params if param.grad is not None))
    if clip_coef < 1.0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            amp_C.multi_tensor_scale, dummy_overflow_buf, [
                grads, grads], clip_coef
        )
    return total_norm
