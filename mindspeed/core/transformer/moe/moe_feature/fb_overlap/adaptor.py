# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from logging import getLogger
from typing import Optional
from functools import wraps
import torch
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.training.utils import get_args
from .modules.experts import MindSpeedFbOverlapGmmExperts
from .modules.shared_experts import SharedExpertMLPFbOverlap
from .modules.moe_layer import MindSpeedFbOverlapMoELayer


def _make_backward_post_hook(self, param: torch.nn.Parameter):
    """
    Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
    ready (i.e., when all grads in a bucket have been computed in all microbatches
    in a batch).
    """

    def hook(*unused):
        if param in self.param_to_bucket_group:
            if not getattr(param, 'skip_grad_accum', False):
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

            if self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param)

        if getattr(param, 'skip_grad_accum', False):
            param.skip_grad_accum = False

    return hook


def get_moe_module_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        spec = fn(*args, **kwargs)
        args = get_args()
        if args.moe_fb_overlap:
            log = getLogger(__name__)
            log.info("moe_fb_overlap is enabled. Replacing default megatron layer spec for moe...")
            spec.module = MindSpeedFbOverlapMoELayer
            spec.submodules.experts.module = MindSpeedFbOverlapGmmExperts
            spec.submodules.shared_experts.module = SharedExpertMLPFbOverlap

        return spec

    return wrapper
