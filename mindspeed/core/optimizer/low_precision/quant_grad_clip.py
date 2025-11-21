# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024; NVIDIA CORPORATION.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

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

from megatron.core.utils import to_local_if_dtensor


def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    float_grads: List[torch.Tensor] = []
    quant_grads: List[torch.Tensor] = []
    for grad in grads_for_norm:
        if grad is None:
            continue
        if hasattr(grad, 'meta') and getattr(grad, 'meta', None) is not None:
            quant_grads.append(grad)
        else:
            float_grads.append(grad)

    norm_type = float(norm_type)
    total_norm = 0.0

    if norm_type == inf:
        if float_grads:
            total_norm = max(float(grad.abs().max()) for grad in float_grads)
        if quant_grads:
            quant_norm = max(
                (grad.float() * grad.meta.scale_inv[0]).abs().max()
                for grad in quant_grads
            )
            total_norm = max(total_norm, quant_norm)
        total_norm_tensor = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=grad_stats_parallel_group,
        )
        total_norm = total_norm_tensor[0].item()
        return total_norm

    if norm_type == 2.0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        if float_grads:
            grad_norm, _ = multi_tensor_applier(
                l2_norm_impl,
                dummy_overflow_buf,
                [float_grads],
                False,
            )
        else:
            grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
        total_norm += float(grad_norm**norm_type)
    else:
        for grad in float_grads:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += float(grad_norm**norm_type)

    for local_grad in quant_grads:
        grad_norm = torch.norm(local_grad.float() * local_grad.meta.scale_inv[0], norm_type)
        total_norm += float(grad_norm**norm_type)

    total_norm_tensor = torch.tensor([total_norm], dtype=torch.float, device='cuda')
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_norm_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=grad_stats_parallel_group,
    )
    total_norm = total_norm_tensor.item() ** (1.0 / norm_type)

    return total_norm


def clip_grad_by_total_norm_fp32_wrapper(func):
    @wraps(func)
    def clip_grad_by_total_norm_fp32(
        parameters: Union[List[torch.Tensor], torch.Tensor],
        max_norm: Union[int, float],
        total_norm: float,
        use_decoupled_grad: bool = False,
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        quant_scale_invs: List[torch.Tensor] = []
        if not use_decoupled_grad:
            for param in parameters:
                quant_grad = getattr(param, 'quant_grad', None)
                if quant_grad is None:
                    continue
                if hasattr(quant_grad, 'meta') and getattr(quant_grad, 'meta', None) is not None:
                    scale_inv = getattr(quant_grad.meta, 'scale_inv', None)
                    if scale_inv is None or not torch.is_tensor(scale_inv):
                        continue
                    if not torch.isfinite(scale_inv).all():
                        print("Skipping quant scale_inv with non-finite entries during clipping.")
                        continue
                    quant_scale_invs.append(scale_inv)

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
        else:
            if quant_scale_invs and total_norm > 0.0:
                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
                    multi_tensor_applier(
                        multi_tensor_scale_impl,
                        dummy_overflow_buf,
                        [quant_scale_invs, quant_scale_invs],
                        clip_coef,
                    )
        return func(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)
    return clip_grad_by_total_norm_fp32