# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import warnings

from typing import Callable, Optional, List, Tuple, Dict
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
from megatron.training.utils import print_rank_0
from megatron.core import mpu
from mindspeed.core.distributed.layerzero.debug.sum import all_gather_into_flat_tensor, print_total_grad_sum
from .sharded_grad_scaler import ShardedGradScaler
from .clip import clip_grad_norm


def _get_param_groups(
    model_chunks: List,
    no_weight_decay_cond: Callable,
    scale_lr_cond: Callable,
    lr_mult: float,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.

    Returns:
        List of parameter groups.
    """
    if not isinstance(model_chunks, list):
        model_chunks = [model_chunks]
    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                #! currently do not support norm parameters, case all zero1 param has len(param.shape) == 1
                no_wd = name.endswith(".bias") or getattr(param, "_is_1D_param", False)

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, lr_mult = 0.0, 1.0
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            key = (wd_mult, lr_mult)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    param_groups = []
    for (wd_mult, lr_mult), params in params_map.items():
        if len(params) == 0:
            raise ValueError(f"Empty params list")
        param_groups.append(
            {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': lr_mult,
                'is_decoupled_lr' : False
            }
        )
    return param_groups


def get_optimizer(
    config: OptimizerConfig,
    model: List,
    no_weight_decay_cond: Callable = None,
    scale_lr_cond: Callable = None,
    lr_mult: float = 1.0
) -> "MegatronOptimizer":  
    param_groups = _get_param_groups(model, no_weight_decay_cond, scale_lr_cond, lr_mult)
    optimizer = _get_zero_optimizer(config, param_groups)
    return optimizer


def _get_zero_optimizer(
    config,
    param_groups
):
    print(f"{config.weight_decay=}")
    if config.optimizer == 'adam':
        optimizer = Adam(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
        init_state_fn = None

    elif config.optimizer == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
        init_state_fn = None
    else:
        raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    
    grad_scaler = None
    if config.fp16:
        grad_scaler = ShardedGradScaler(
            init_scale=config.initial_loss_scale,
            min_scale=config.min_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=config.loss_scale_window,
            hysteresis=config.hysteresis,
        )

    optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
    optimizer = LayerZeROptimizer(*optimizer_args)
    return optimizer


def pp_stages():
    if not mpu.is_initialized():
        return 1
    world_size = dist.get_world_size()
    return world_size // len(mpu.get_pipeline_model_parallel_group())


def pp_broadcast_grad_scale(grad_scale, device):
    if pp_stages() == 1:
        return grad_scale
    pp_world_size = mpu.get_pipeline_model_parallel_world_size()
    world_size = dist.get_world_size()
    last_stage_rank0 = world_size - pp_world_size
    if not isinstance(grad_scale, torch.Tensor):
        grad_scale = torch.tensor(grad_scale, dtype=torch.float32).to(device)
    dist.broadcast(grad_scale, src=last_stage_rank0)
    return grad_scale


class LayerZeROptimizer(MegatronOptimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[ShardedGradScaler],
        init_state_fn: Callable = lambda x: None,
        process_group: Optional[ProcessGroup] = dist.group.WORLD,
    ):
        super().__init__(optimizer, config, lambda x: None)
        self.grad_scaler = grad_scaler
        self.process_group = process_group or dist.group.WORLD
        self.device = torch.device('cuda')
        
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Simple scaling."""
        return self.get_loss_scale() * loss
    
    def get_loss_scale(self) -> torch.Tensor:
        '''if PP enabled, broadcast scale'''
        if self.grad_scaler is None:
            return torch.tensor([1.], dtype=torch.float32, device=self.device)
        return self.grad_scaler.loss_scale.to(self.device)
    
    @torch.no_grad()
    def step(self) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        if self.grad_scaler:
            self.grad_scaler._scale = pp_broadcast_grad_scale(self.get_loss_scale(), self.device)
            found_inf = self.grad_scaler.unscale_(self.optimizer)
        else:
            found_inf = False
        
        grad_norm = None
        if self.config.clip_grad > 0.0:
            if self.process_group is None:
                raise RuntimeError(f"{self.process_group=} is None")
            grad_norm = clip_grad_norm(self.get_parameters(), self.config.clip_grad, norm_type=2, process_group=self.process_group)
        
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None

        if self.grad_scaler:
            self.grad_scaler._meg_step(self.optimizer)
            self.grad_scaler.update()  
        else:
            self.optimizer.step()
            
        return not found_inf, grad_norm, num_zeros_in_grad
        
    def prepare_grads(self) -> bool:
        raise RuntimeError("This function should not be explicitly called by user")
    
    def step_with_ready_grads(self) -> bool:
        raise RuntimeError("This function should not be explicitly called by user")
    
    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        raise RuntimeError("This function should not be explicitly called by user")
    
    def count_zeros(self):
        num_zeros = sum(param.grad.numel() - torch.count_nonzero(param.grad) \
            for param in self.get_parameters() if param.grad is not None)
        dist.all_reduce(num_zeros, group=self.process_group)
        return num_zeros
    
    def reload_model_params(self):
        '''Megatron optimizer api'''
        pass
    
    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
        self.optimizer.load_state_dict(state_dict[optimizer_key])
        # Grad scaler.
        if self.grad_scaler:
            if "grad_scaler" not in state_dict:
                warnings.warn(f"grad scaler state dict missing")
            else:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
   
    def sharded_state_dict(
        self, model_sharded_state_dict, is_loading: bool = False
        ):
        """ Builds sharded state dict for the optimizer, based on model's sharded state dict.
        
        Args:
            model_sharded_state_dict (ShardedStateDict): sharded state dict of the model
            is_loading (bool, optional): flag indicating whether the state dict will be used to save or load the optimizer state.
                Defaults to False.
        
        Returns: optimizer sharded state dict
        """
        raise NotImplementedError("This api should not be called")
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad()
        
    def disable_pre_hook(self):
        return

    def enable_pre_hook(self):
        return
        