# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Dict
import logging

from megatron.core.distributed.distributed_data_parallel import logger

from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.utils import log_single_rank
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
import torch


def _make_backward_post_hook(self, param: torch.nn.Parameter):
    """
    Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
    ready (i.e., when all grads in a bucket have been computed in all microbatches
    in a batch).
    """

    def hook(*unused):
        if is_graph_capturing():
            return

        if param in self.param_to_bucket_group:
            assert param.requires_grad
            if self.ddp_config.overlap_grad_reduce:
                assert (
                    param.grad is not None
                ), 'param.grad being None is not safe when overlap_grad_reduce is True'
            if param.grad is not None and (
                not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
            ):
                param.main_grad.add_(*unused)
            param.grad = None

            if self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param)
            
            if hasattr(param, "main_grad"):
                return param.main_grad
            return param.grad

    return hook
