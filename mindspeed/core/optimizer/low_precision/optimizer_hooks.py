from functools import wraps
from typing import Iterable, List, Optional

import torch
import torch.distributed

from megatron.training import get_args

_BASIC_TOKEN_TO_DTYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': torch.uint8,
}
_QUANT_GRAD_DTYPE_ALIAS = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
}


def _resolve_dtype(value, default: torch.dtype = torch.float32) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return _BASIC_TOKEN_TO_DTYPE.get(value.lower(), default)
    return default


def _sync_precision_aware_config(config, args_namespace) -> None:
    config.main_grads_dtype = _resolve_dtype(getattr(args_namespace, 'main_grads_dtype', torch.float32))
    config.main_params_dtype = _resolve_dtype(getattr(args_namespace, 'main_params_dtype', torch.float32))
    config.exp_avg_dtype = _resolve_dtype(getattr(args_namespace, 'exp_avg_dtype', torch.float32))
    config.exp_avg_sq_dtype = _resolve_dtype(getattr(args_namespace, 'exp_avg_sq_dtype', torch.float32))


def _is_precision_enabled(args_namespace) -> bool:
    return getattr(args_namespace, 'use_precision_aware_optimizer', False)


def _is_quant_enabled(args_namespace) -> bool:
    return getattr(args_namespace, 'use_quant_optimizer', False)


def _cast_quant_grad_if_needed(grad_tensor: torch.Tensor, dtype_token: Optional[str]):
    if dtype_token is None:
        return grad_tensor
    target_dtype = _QUANT_GRAD_DTYPE_ALIAS.get(dtype_token)
    if target_dtype is None:
        return grad_tensor
    if hasattr(grad_tensor, 'meta') and getattr(grad_tensor, 'meta', None) is not None:
        return grad_tensor
    if grad_tensor.dtype == target_dtype:
        return grad_tensor
    return grad_tensor.to(dtype=target_dtype, non_blocking=True)


def optimizer_config_init_wrapper(init_func):
    @wraps(init_func)
    def optimizer_config_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args_namespace = get_args()
        if _is_precision_enabled(args_namespace):
            _sync_precision_aware_config(self, args_namespace)
        self.reuse_fp32_param = getattr(args_namespace, 'reuse_fp32_param', False)
        self.use_quant_optimizer = _is_quant_enabled(args_namespace)

    return optimizer_config_init


def optimizer_config_post_init_wrapper(post_init_func):
    @wraps(post_init_func)
    def optimizer_config_post_init(*args, **kwargs):
        self = args[0]
        args_namespace = get_args()
        if _is_precision_enabled(args_namespace) or _is_quant_enabled(args_namespace):
            if self.optimizer != 'adam':
                raise AssertionError('MindSpeed low precision optimizers only support Adam.')
            if not self.use_distributed_optimizer:
                raise AssertionError('MindSpeed low precision optimizers require distributed optimizer.')
            if self.optimizer_cpu_offload:
                raise AssertionError('MindSpeed low precision optimizers do not support optimizer CPU offload.')
            return
        return post_init_func(*args, **kwargs)

    return optimizer_config_post_init


def _build_mindspeed_precision_aware_optimizer(
    config,
    model_chunks,
    param_groups,
    per_model_buffers=None,
    model_parallel_group=None,
    data_parallel_group=None,
    data_parallel_group_gloo=None,
    data_parallel_group_idx=None,
    distributed_optimizer_instance_id=0,
):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )
    from mindspeed.core.optimizer.low_precision import precision_aware_adamw

    if param_groups:
        optimizer = precision_aware_adamw.AdamW(
            params=param_groups,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        init_state_fn = None
    else:
        optimizer = None
        init_state_fn = None

    grad_scaler = None
    if config.loss_scale:
        grad_scaler = ConstantGradScaler(config.loss_scale)
    elif config.fp16:
        grad_scaler = DynamicGradScaler(
            initial_scale=config.initial_loss_scale,
            min_scale=config.min_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=config.loss_scale_window,
            hysteresis=config.hysteresis,
        )

    optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
    if config.use_distributed_optimizer:
        optimizer = DistributedOptimizer(
            *optimizer_args,
            model_chunks=model_chunks,
            per_model_buffers=per_model_buffers,
            data_parallel_group=data_parallel_group,
            data_parallel_group_gloo=data_parallel_group_gloo,
            data_parallel_group_idx=data_parallel_group_idx,
            distributed_optimizer_instance_id=distributed_optimizer_instance_id,
        )
    elif config.fp16 or config.bf16:
        optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    return optimizer


def _build_mindspeed_quant_optimizer(
    config,
    model_chunks,
    param_groups,
    per_model_buffers=None,
    model_parallel_group=None,
    data_parallel_group=None,
    data_parallel_group_gloo=None,
    data_parallel_group_idx=None,
    distributed_optimizer_instance_id=0,
):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )
    from mindspeed.core.optimizer.low_precision import quant_adamw

    if param_groups:
        optimizer = quant_adamw.AdamW(
            params=param_groups,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        init_state_fn = None
    else:
        optimizer = None
        init_state_fn = None

    grad_scaler = None
    if config.loss_scale:
        grad_scaler = ConstantGradScaler(config.loss_scale)
    elif config.fp16:
        grad_scaler = DynamicGradScaler(
            initial_scale=config.initial_loss_scale,
            min_scale=config.min_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=config.loss_scale_window,
            hysteresis=config.hysteresis,
        )

    optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
    if config.use_distributed_optimizer:
        optimizer = DistributedOptimizer(
            *optimizer_args,
            model_chunks=model_chunks,
            per_model_buffers=per_model_buffers,
            data_parallel_group=data_parallel_group,
            data_parallel_group_gloo=data_parallel_group_gloo,
            data_parallel_group_idx=data_parallel_group_idx,
            distributed_optimizer_instance_id=distributed_optimizer_instance_id,
        )
    elif config.fp16 or config.bf16:
        optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    return optimizer


def get_optimizer_builder_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_namespace = get_args()
        if _is_precision_enabled(args_namespace):
            return _build_mindspeed_precision_aware_optimizer(*args, **kwargs)
        if _is_quant_enabled(args_namespace):
            return _build_mindspeed_quant_optimizer(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def _iter_param_groups(optimizer, group_names: Iterable[str]):
    for name in group_names:
        groups = getattr(optimizer, name, None)
        if not groups:
            continue
        for group in groups:
            yield group


def _cast_grad(grad: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    tensor = grad
    if hasattr(tensor, 'to_local'):
        tensor = tensor.to_local()
    if tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    else:
        tensor = tensor.clone()
    return tensor


def collect_main_grad_data_for_unscaling_wrapper(func):
    @wraps(func)
    def _collect_main_grad_data_for_unscaling(self):
        args_namespace = get_args()
        precision_enabled = _is_precision_enabled(args_namespace)
        quant_enabled = _is_quant_enabled(args_namespace)

        if precision_enabled:
            main_grads = []
            group_names = ['fp32_from_float16_groups', 'fp32_from_fp32_groups']
            grad_dtype = _resolve_dtype(getattr(args_namespace, 'main_grads_dtype', torch.float32))
            for group in _iter_param_groups(self, group_names):
                for param in group:
                    grad = getattr(param, 'decoupled_grad', None)
                    if grad is None:
                        grad = getattr(param, 'main_grad', None)
                    if grad is None:
                        continue
                    main_grads.append(_cast_grad(grad, grad_dtype))
            return main_grads

        if quant_enabled:
            main_grads = func(self)
            if getattr(args_namespace, 'quant_grads', False) and not getattr(self, 'is_stub_optimizer', False):
                meta_grads_scale_inv = []
                main_groups = list(getattr(self, 'fp32_from_float16_groups', []))
                fallback_groups = list(getattr(self, 'float16_groups', []))
                for group_idx, main_group in enumerate(main_groups):
                    fallback_group = fallback_groups[group_idx] if group_idx < len(fallback_groups) else None
                    for param_idx, main_param in enumerate(main_group):
                        target_param = main_param
                        if target_param is None and fallback_group is not None and param_idx < len(fallback_group):
                            target_param = fallback_group[param_idx]
                        if target_param is None:
                            continue
                        quant_grad = getattr(target_param, 'quant_grad', None)
                        if quant_grad is not None and getattr(quant_grad, 'meta', None) is not None:
                            meta_grads_scale_inv.append(quant_grad.meta.scale_inv)
                if meta_grads_scale_inv:
                    return main_grads, meta_grads_scale_inv
            return main_grads

        return func(self)

    return _collect_main_grad_data_for_unscaling


def copy_model_grads_to_main_grads_wrapper(func):
    @wraps(func)
    def _copy_model_grads_to_main_grads(self):
        args_namespace = get_args()
        precision_enabled = _is_precision_enabled(args_namespace)
        quant_enabled = _is_quant_enabled(args_namespace)

        if quant_enabled:
            if getattr(self, 'is_stub_optimizer', False):
                return func(self)
            if not getattr(args_namespace, 'quant_grads', False):
                return func(self)
            quant_dtype_token = getattr(args_namespace, 'quant_grads_dtype', None)
            float16_groups = list(getattr(self, 'float16_groups', []))
            fp32_from_float16_groups = list(getattr(self, 'fp32_from_float16_groups', []))
            for model_group, main_group in zip(float16_groups, fp32_from_float16_groups):
                for model_param, main_param in zip(model_group, main_group):
                    grad_source = getattr(model_param, 'main_grad', None) or model_param.grad
                    if grad_source is None:
                        continue
                    quant_grad = _cast_quant_grad_if_needed(grad_source, quant_dtype_token)
                    target_param = main_param if main_param is not None else model_param
                    target_param.quant_grad = quant_grad
                    target_param.grad = None
                    model_param.grad = None
            for model_group in list(getattr(self, 'fp32_from_fp32_groups', [])):
                for model_param in model_group:
                    main_grad = getattr(model_param, 'main_grad', None)
                    if main_grad is None:
                        continue
                    model_param.quant_grad = _cast_quant_grad_if_needed(main_grad, quant_dtype_token)
                    model_param.grad = None
            return

        if precision_enabled:
            grad_dtype = _resolve_dtype(getattr(args_namespace, 'main_grads_dtype', torch.float32))
            if grad_dtype == torch.float32:
                return func(self)
            for group in _iter_param_groups(self, ['fp32_from_float16_groups', 'fp32_from_fp32_groups']):
                for param in group:
                    grad = param.grad
                    if grad is None:
                        continue
                    param.decoupled_grad = grad.to(dtype=grad_dtype)
                    param.grad = None
            return

        return func(self)

    return _copy_model_grads_to_main_grads


def unscale_main_grads_and_check_for_nan(self):
    collected = self._collect_main_grad_data_for_unscaling()
    if isinstance(collected, tuple):
        main_grads, meta_grads_scale_inv = collected
    else:
        main_grads = collected
        meta_grads_scale_inv = []
    self.found_inf.fill_(0.0)
    if main_grads:
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale
        )
    if meta_grads_scale_inv:
        torch._amp_foreach_non_finite_check_and_unscale_(
            meta_grads_scale_inv, self.found_inf, self.grad_scaler.inv_scale
        )
    torch.distributed.all_reduce(
        self.found_inf,
        op=torch.distributed.ReduceOp.MAX,
        group=self.get_grad_stats_parallel_group(),
    )
    return self.found_inf.item() > 0


def get_main_grads_for_grad_norm(self):
    args_namespace = get_args()
    precision_enabled = _is_precision_enabled(args_namespace)
    quant_enabled = _is_quant_enabled(args_namespace)
    params = self.get_parameters()
    grads_for_norm = []
    for param in params:
        grad = None
        if quant_enabled and hasattr(param, 'quant_grad') and param.quant_grad is not None:
            grad = param.quant_grad
        elif precision_enabled:
            grad = getattr(param, 'decoupled_grad', None) or param.grad
        else:
            grad = param.grad
        if grad is None:
            continue
        grads_for_norm.append(grad)
    return grads_for_norm


def zero_grad_group_helper_wrapper(func):
    @wraps(func)
    def _zero_grad_group_helper(group: List[torch.nn.Parameter], set_to_none: bool, use_decoupled_grad: bool = False):
        func(group, set_to_none, use_decoupled_grad)
        args_namespace = get_args()
        precision_enabled = _is_precision_enabled(args_namespace)
        quant_enabled = _is_quant_enabled(args_namespace)

        if precision_enabled and use_decoupled_grad:
            for param in group:
                decoupled_grad = getattr(param, 'decoupled_grad', None)
                if decoupled_grad is None:
                    continue
                if set_to_none:
                    param.decoupled_grad = None
                else:
                    if decoupled_grad.grad_fn is not None:
                        decoupled_grad = decoupled_grad.detach()
                    else:
                        decoupled_grad.requires_grad_(False)
                    decoupled_grad.zero_()
                    param.decoupled_grad = decoupled_grad

        if quant_enabled:
            for param in group:
                quant_grad = getattr(param, 'quant_grad', None)
                if quant_grad is None:
                    continue
                if set_to_none:
                    param.quant_grad = None
                else:
                    if quant_grad.grad_fn is not None:
                        quant_grad = quant_grad.detach()
                    else:
                        quant_grad.requires_grad_(False)
                    quant_grad.zero_()
                    param.quant_grad = quant_grad

    return _zero_grad_group_helper


def prepare_grads_impl(self) -> bool:
    #print('prepare_grads_impl'+'S0'*300)
    timers = self.config.timers
    if timers is not None:
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    if not getattr(self, 'is_stub_optimizer', False):
        self._copy_model_grads_to_main_grads()
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

    if getattr(self.config, 'reuse_fp32_param', False) and not getattr(self, 'is_stub_optimizer', False):
        self.fp16_tensor_convert_to_fp32_tensor()

    if self.grad_scaler:
        if timers is not None:
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        if timers is not None:
            timers('optimizer-unscale-and-check-inf').stop()
        self.grad_scaler.update(found_inf_flag)
        return found_inf_flag

    return False


def step_with_ready_grads_impl(self) -> bool:
    print('step_with_ready_grads_impl'+'S1'*300)
    timers = self.config.timers
    if timers is not None:
        timers('optimizer-inner-step', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    if not getattr(self, 'is_stub_optimizer', False):
        self.optimizer.step()
    if timers is not None:
        timers('optimizer-inner-step').stop()

    if timers is not None:
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    if not getattr(self, 'is_stub_optimizer', False):
        if getattr(self.config, 'reuse_fp32_param', False):
            self.fp32_tensor_convert_to_fp16_tensor()
        else:
            self._copy_main_params_to_model_params()
    if timers is not None:
        timers('optimizer-copy-main-to-model-params').stop()
    return True


def mixed_precision_optimizer_step_impl(self):
    print('mixed_precision_optimizer_step_impl'+'S2'*300)
    timers = self.config.timers
    timers('optimizer-copy-to-main-grad', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    if not getattr(self, 'is_stub_optimizer', False):
        self._copy_model_grads_to_main_grads()
    timers('optimizer-copy-to-main-grad').stop()

    if getattr(self.config, 'reuse_fp32_param', False) and not getattr(self, 'is_stub_optimizer', False):
        self.fp16_tensor_convert_to_fp32_tensor()

    if self.grad_scaler:
        timers('optimizer-unscale-and-check-inf', log_level=1).start(
            barrier=self.config.barrier_with_L1_time)
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        timers('optimizer-unscale-and-check-inf').stop()
        self.grad_scaler.update(found_inf_flag)
        if found_inf_flag:
            return False, None, None

    timers('optimizer-clip-main-grad', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    grad_norm = None
    if self.config.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.config.clip_grad)
    timers('optimizer-clip-main-grad').stop()

    timers('optimizer-count-zeros', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
    timers('optimizer-count-zeros').stop()

    timers('optimizer-inner-step', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    if not getattr(self, 'is_stub_optimizer', False):
        self.optimizer.step()
    timers('optimizer-inner-step').stop()

    timers('optimizer-copy-main-to-model-params', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    if not getattr(self, 'is_stub_optimizer', False):
        if getattr(self.config, 'reuse_fp32_param', False):
            self.fp32_tensor_convert_to_fp16_tensor()
        else:
            self._copy_main_params_to_model_params()
    timers('optimizer-copy-main-to-model-params').stop()

    return True, grad_norm, num_zeros_in_grad
