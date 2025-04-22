# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Callable, List, Optional
import logging

import torch

from megatron.core.optimizer import logger, _get_param_groups, _update_min_and_max_lr_in_param_groups, _get_megatron_optimizer_based_on_param_groups

from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.optimizer.optimizer import MegatronOptimizer, ChainedOptimizer

from megatron.core.utils import log_single_rank
from megatron.core import mpu


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.

    Returns:
        Instance of MegatronOptimizer.
    """

    log_single_rank(logger, logging.INFO, 'Setting up optimizer with config %s', config)

    # Collect param groups.
    param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_decoupled_learning_rate=config.decoupled_lr is not None,
    )
    param_groups = _update_min_and_max_lr_in_param_groups(
        param_groups,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    # Fake params to construct optmizer
    if not param_groups:
        fake_params = torch.zeros([1,], dtype=torch.float, requires_grad=True)
        fake_params.fake = True
        fake_params.grad = fake_params.clone()
        fake_params.main_grad = fake_params.clone()
        param_groups.append({'params': fake_params, 'wd_mult': 0.0, 'lr_mult': 0.0, 'is_decoupled_lr': False})

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}
    per_model_ep_buffers = {}
    for model_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, 'buffers'):
            per_model_buffers[model_idx] = model_chunk.buffers
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())
    optimizers = [
        _get_megatron_optimizer_based_on_param_groups(
            config,
            param_groups=dense_param_groups,
            per_model_buffers=per_model_buffers,
            model_parallel_group=mpu.get_model_parallel_group(),
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
            data_parallel_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
            data_parallel_group_idx=model_parallel_rank,
        )
    ]
    if moe_param_groups:
        model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
        expert_parallel_rank = mpu.get_expert_model_parallel_rank()
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                param_groups=moe_param_groups,
                per_model_buffers=per_model_ep_buffers,
                model_parallel_group=mpu.get_model_parallel_group(with_expert_parallel=True),
                data_parallel_group=mpu.get_data_modulo_expert_parallel_group(
                    with_context_parallel=True
                ),
                data_parallel_group_gloo=mpu.get_data_modulo_expert_parallel_group_gloo(
                    with_context_parallel=True
                ),
                data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size
                + model_parallel_rank,
            )
        )

    if len(optimizers) == 1:
        return optimizers[0]

    return ChainedOptimizer(optimizers)