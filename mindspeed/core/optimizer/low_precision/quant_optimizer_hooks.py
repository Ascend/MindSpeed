import os
from functools import wraps
from typing import List
import logging
import types

import torch

import megatron.core.tensor_parallel as tensor_parallel
from megatron.training import get_args
from megatron.core.transformer.module import param_is_not_shared


def _global_world_size() -> int:
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    try:
        return torch.distributed.get_world_size()
    except RuntimeError:
        return 1

logger = logging.getLogger(__name__)
_PATCHED_ADAM_CACHE = {}


@torch.no_grad()
def prepare_grads_impl(self) -> bool:
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
        if getattr(self.config, 'reuse_fp32_param', False):
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


@torch.no_grad()
def step_with_ready_grads_impl(self) -> bool:
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


@torch.no_grad()
def mixed_precision_optimizer_step_impl(self):
    timers = self.config.timers
    timers('optimizer-copy-to-main-grad', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    if not getattr(self, 'is_stub_optimizer', False):
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()
    if getattr(self.config, 'reuse_fp32_param', False) and not getattr(self, 'is_stub_optimizer', False):
        if getattr(self.config, 'reuse_fp32_param', False):
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
    num_zeros_in_grad = self.count_zeros() if \
        self.config.log_num_zeros_in_grad else None
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


def optimizer_config_init_wrapper(init_func):
    @wraps(init_func)
    def optimizer_config_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args_namespace = get_args()
        quant_enabled = getattr(args_namespace, 'use_quant_optimizer', False)
        self.reuse_fp32_param = getattr(args_namespace, 'reuse_fp32_param', False)
        self.use_quant_optimizer = quant_enabled
    return optimizer_config_init


def get_megatron_optimizer_func_wrapper(func):
    @wraps(func)
    def get_megatron_optimizer_func(*args, **kwargs):
        chained_optimizer = func(*args, **kwargs)
        args_namespace = get_args()
        if hasattr(chained_optimizer, "chained_optimizers"):
            for optim in chained_optimizer.chained_optimizers:
                if hasattr(optim, "optimizer") and optim.optimizer is not None:
                    optim.optimizer.ema_decay = getattr(args_namespace, 'ema_decay', None)
                else:
                    setattr(optim, 'ema_decay', getattr(args_namespace, 'ema_decay', None))
            return chained_optimizer
        if hasattr(chained_optimizer, "optimizer") and chained_optimizer.optimizer is not None:
            chained_optimizer.optimizer.ema_decay = getattr(args_namespace, 'ema_decay', None)
        return chained_optimizer
    return get_megatron_optimizer_func


def optimizer_config_post_init_wrapper(post_init_func):
    @wraps(post_init_func)
    def optimizer_config_post_init(*args, **kwargs):
        self = args[0]
        args_namespace = get_args()
        if getattr(args_namespace, 'use_quant_optimizer', False):
            if self.optimizer != 'adam':
                raise AssertionError('MindSpeed quant optimizer only supports Adam.')
            if not self.use_distributed_optimizer:
                raise AssertionError('MindSpeed quant optimizer requires distributed optimizer.')
            if self.optimizer_cpu_offload:
                raise AssertionError('MindSpeed quant optimizer does not support optimizer CPU offload.')
        else:
            post_init_func(*args, **kwargs)
        return None
    return optimizer_config_post_init


def get_optimizer_builder_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        config = args[0]
        args_namespace = get_args()
        if getattr(args_namespace, 'use_quant_optimizer', False):
            return _build_mindspeed_quant_optimizer(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


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
    from mindspeed.core.optimizer.low_precision import quant_adamw
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
    from megatron.core.optimizer.optimizer import (
        Float16OptimizerWithFloat16Params,
        FP32Optimizer,
    )

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

    config.use_quant_optimizer = True

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


def _get_patched_adam(original_adam, quant_class):
    cached = _PATCHED_ADAM_CACHE.get(original_adam)
    if cached is not None:
        return cached

    adam_meta = type(original_adam)

    class _MindSpeedAdamMeta(adam_meta):
        def __instancecheck__(self, instance):  # type: ignore[override]
            if adam_meta.__instancecheck__(self, instance):  # type: ignore[misc]
                return True
            return isinstance(instance, quant_class)

    patched = _MindSpeedAdamMeta(
        f"{original_adam.__name__}MindSpeedProxy",
        (original_adam,),
        {},
    )
    _PATCHED_ADAM_CACHE[original_adam] = patched
    return patched


def distributed_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def _wrapper(self, optimizer, config, grad_scaler, init_state_fn, *args, **kwargs):
        patched = False
        original_adam = init_func.__globals__.get('Adam')
        if optimizer is not None and original_adam is not None:
            try:
                from mindspeed.core.optimizer.low_precision import quant_adamw
            except ImportError:
                quant_adamw = None
            if quant_adamw is not None and isinstance(optimizer, quant_adamw.AdamW):
                patched_adam = _get_patched_adam(original_adam, quant_adamw.AdamW)
                init_func.__globals__['Adam'] = patched_adam
                patched = True
        try:
            return init_func(self, optimizer, config, grad_scaler, init_state_fn, *args, **kwargs)
        finally:
            if patched:
                init_func.__globals__['Adam'] = original_adam

    return _wrapper


def reuse_fp32_param_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args_namespace = get_args()
        self.reuse_fp32_param = getattr(args_namespace, 'reuse_fp32_param', False)
        if not self.reuse_fp32_param:
            return
        self.res_float16_groups = []
        self.float16_float32_groups = []
        self.int32_float32_groups = []
        for float16_params_this_group, fp32_from_float16_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            res_float16_params_this_group = []
            float16_float32_params_this_group = []
            int32_float32_params_this_group = []
            for i, (_, fp32_from_fp16_param) in enumerate(zip(float16_params_this_group, fp32_from_float16_group)):
                res_float16_params_this_group.append(
                    torch.empty((fp32_from_fp16_param.numel() * 1), dtype=torch.bfloat16, device=fp32_from_fp16_param.device))
                float16_float32_params_this_group.append(
                    torch.empty((fp32_from_fp16_param.numel() * 2), dtype=torch.bfloat16, device=fp32_from_fp16_param.device))
                int32_float32_params_this_group.append(
                    torch.empty((fp32_from_fp16_param.numel() * 1), dtype=torch.int32, device=fp32_from_fp16_param.device))
                init_and_reuse_storage_of_tensors(fp32_from_float16_group[i],
                                                  float16_float32_params_this_group[-1],
                                                  res_float16_params_this_group[-1],
                                                  float16_params_this_group[i],
                                                  int32_float32_params_this_group[-1])
            self.res_float16_groups.append(res_float16_params_this_group)
            self.float16_float32_groups.append(float16_float32_params_this_group)
            self.int32_float32_groups.append(int32_float32_params_this_group)
        self._copy_model_params_to_main_params = _copy_model_params_to_main_params
        if getattr(args_namespace, 'npu_deterministic', False):
            self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(fp16_tensor_convert_to_fp32_tensor_deterministic, self)
            self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(fp32_tensor_convert_to_fp16_tensor_deterministic, self)
        else:
            self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(fp16_tensor_convert_to_fp32_tensor, self)
            self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(fp32_tensor_convert_to_fp16_tensor, self)
    return reuse_fp32_param_init


def _copy_model_params_to_main_params():
    pass


def init_and_reuse_storage_of_tensors(
        fp32_tensor,
        bf16_fp32_tensor,
        res_tensor,
        bf16_tensor,
        int32_tensor
):
    from mindspeed.op_builder import AlgorithmOpBuilder
    reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
    reuse_data_ptr(bf16_fp32_tensor, fp32_tensor, 0)
    reuse_data_ptr(int32_tensor, fp32_tensor, 0)
    fp32_tensors_to_bf16_tensors([int32_tensor], [bf16_fp32_tensor])
    reuse_data_ptr(res_tensor, bf16_fp32_tensor, 0)
    reuse_data_ptr(bf16_tensor, bf16_fp32_tensor, res_tensor.numel())


def fp16_tensor_convert_to_fp32_tensor(self):
    for int32_float32_group, float16_param_group in zip(
            self.int32_float32_groups, self.float16_float32_groups):
        bf16_tensors_to_fp32_tensors(int32_float32_group, float16_param_group)


def fp32_tensor_convert_to_fp16_tensor(self):
    for int32_float32_param_group, float16_param_group in zip(
        self.int32_float32_groups, self.float16_float32_groups):
        fp32_tensors_to_bf16_tensors(int32_float32_param_group, float16_param_group)


def fp32_tensors_to_bf16_tensors(int32_tensors, bf16_fp32_tensors):
    for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            return
        int32_tensor.add_(32768)
        bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())


def bf16_tensors_to_fp32_tensors(int32_tensors, bf16_fp32_tensors):
    for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            return
        bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
        int32_tensor.sub_(32768)


def fp16_tensor_convert_to_fp32_tensor_deterministic(self):
    for int32_float32_group, float16_param_group, fp32_from_float16_group in zip(
        self.int32_float32_groups, self.float16_float32_groups, self.fp32_from_float16_groups):
        bf16_tensors_to_fp32_tensors_deterministic(int32_float32_group, float16_param_group, fp32_from_float16_group, self.optimizer)


def fp32_tensor_convert_to_fp16_tensor_deterministic(self):
    for int32_float32_param_group, float16_param_group, fp32_from_float16_group in zip(
        self.int32_float32_groups, self.float16_float32_groups, self.fp32_from_float16_groups):
        fp32_tensors_to_bf16_tensors_deterministic(int32_float32_param_group, float16_param_group, fp32_from_float16_group, self.optimizer)


def fp32_tensors_to_bf16_tensors_deterministic(int32_tensors, bf16_fp32_tensors, fp32_tensors, optimizer):
    for int32_tensor, bf16_fp32_tensor, fp32_tensor in zip(int32_tensors, bf16_fp32_tensors, fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            return
        odd_even_tensor = ((int32_tensor & 131071) == 32768).int()
        int32_tensor.add_(32768)
        optimizer_exp_avg_save_sign(optimizer, fp32_tensor, int32_tensor, odd_even_tensor)
        bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())


def bf16_tensors_to_fp32_tensors_deterministic(int32_tensors, bf16_fp32_tensors, fp32_tensors, optimizer):
    for int32_tensor, bf16_fp32_tensor, fp32_tensor in zip(int32_tensors, bf16_fp32_tensors, fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            return
        bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
        optimizer_exp_avg_load_sign(optimizer, fp32_tensor, int32_tensor)
        int32_tensor.sub_(32768)


def optimizer_exp_avg_save_sign(optimizer, fp32_param, int32_tensor, odd_even_tensor):
    if "exp_avg_sq" in optimizer.state[fp32_param]:
        int32_tensor.sub_(odd_even_tensor)
        sign_tensor = torch.sign(odd_even_tensor - 0.5).reshape(optimizer.state[fp32_param]["exp_avg_sq"].shape)
        optimizer.state[fp32_param]["exp_avg_sq"].mul_(sign_tensor)


def optimizer_exp_avg_load_sign(optimizer, fp32_param, int32_tensor):
    if "exp_avg_sq" in optimizer.state[fp32_param]:
        odd_even_tensor = (torch.sign(optimizer.state[fp32_param]["exp_avg_sq"]) > 0).reshape(-1)
        optimizer.state[fp32_param]["exp_avg_sq"].abs_()
        int32_tensor.add_(odd_even_tensor)


def collect_main_grad_data_for_unscaling_wrapper(func):
    @wraps(func)
    def _collect_main_grad_data_for_unscaling(self):
        base = func(self)
        meta_grads_scale_inv = []

        def _register_scale_inv(tensor):
            if tensor is None:
                return
            meta = getattr(tensor, 'meta', None)
            if meta is None:
                return
            scale_inv = getattr(meta, 'scale_inv', None)
            if scale_inv is None:
                return
            meta_grads_scale_inv.append(scale_inv)

        for group in getattr(self, 'fp32_from_float16_groups', []):
            for main_param in group:
                _register_scale_inv(getattr(main_param, 'quant_grad', None))

        return base, meta_grads_scale_inv
    return _collect_main_grad_data_for_unscaling


def copy_model_grads_to_main_grads(self):
    args_namespace = get_args()

    for model_group, main_group in zip(
        self.float16_groups, self.fp32_from_float16_groups
    ):
        for model_param, main_param in zip(model_group, main_group):
            if hasattr(model_param, 'main_grad'):
                if args_namespace.quant_grads:
                    main_param.quant_grad = model_param.main_grad
                else:
                    main_param.grad = model_param.main_grad.float()
            else:
                if model_param.grad is not None:
                    if args_namespace.quant_grads:
                        main_param.quant_grad = model_param.grad
                    else:
                        main_param.grad = model_param.grad.float()

            # Safe to deallocate model's grad/main_grad after copying.
            # (If using contiguous buffers, main_grad's memory should
            # persist and therefore should not be deallocated.)
            model_param.grad = None

    # For fp32 grads, we need to reset the grads to main grad.
    for model_group in self.fp32_from_fp32_groups:
        for model_param in model_group:
            if args.quant_grads:
                model_param.quant_grad = model_param.main_grad
            else:
                model_param.grad = model_param.main_grad


def unscale_main_grads_and_check_for_nan(self):
    if getattr(self, 'is_stub_optimizer', False):
        main_grads = []
        meta_grads_scale_inv = []
    else:
        collected = self._collect_main_grad_data_for_unscaling()
        if isinstance(collected, tuple):
            main_grads, meta_grads_scale_inv = collected
        else:
            main_grads = collected
            meta_grads_scale_inv = []
    self.found_inf.fill_(0.0)
    if not getattr(self, 'is_stub_optimizer', False):
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
    params = self.get_parameters()
    grads_for_norm = []
    for param in params:
        if hasattr(param, 'quant_grad') and param.quant_grad is not None:
            grad = param.quant_grad
        elif self.config.use_quant_optimizer and hasattr(param, 'decoupled_grad'):
            grad = param.decoupled_grad
        else:
            grad = param.grad
        if grad is None:
            continue
        if param_is_not_shared(param) and tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
            # Do not dequantize here to avoid materializing FP32 copies of
            # all grads at once. Return underlying tensors and let
            # get_grad_norm_fp32 handle dequantization per-tensor.
            grads_for_norm.append(grad)
    return grads_for_norm


def zero_grad_group_helper_wrapper(func):
    @wraps(func)
    def _zero_grad_group_helper(group: List[torch.nn.Parameter], set_to_none: bool, use_decoupled_grad: bool = False):
        func(group, set_to_none, use_decoupled_grad)
        if use_decoupled_grad:
            return
        for param in group:
            if hasattr(param, 'quant_grad'):
                if set_to_none:
                    param.quant_grad = None
                else:
                    if param.quant_grad.grad_fn is not None:
                        param.quant_grad.detach_()
                    else:
                        param.quant_grad.requires_grad_(False)
                    param.quant_grad.zero_()
    return _zero_grad_group_helper