import os
import types
from functools import wraps
from typing import List

import torch
from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.core.transformer.module import param_is_not_shared


@torch.no_grad()
def prepare_grads(self) -> bool:
    """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
    timers = self.config.timers

    # Copy gradients from model params to main params.
    if timers is not None:
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    self._copy_model_grads_to_main_grads()
    if timers is not None:
        timers('optimizer-copy-to-main-grad').stop()

    if self.config.reuse_fp32_param:
        # bf16 -> fp32
        self.fp16_tensor_convert_to_fp32_tensor()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if self.grad_scaler:

        # Unscale and check for inf/nan.
        if timers is not None:
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        if timers is not None:
            timers('optimizer-unscale-and-check-inf').stop()

        # We are done with scaling gradients
        # so we can update the loss scale.
        self.grad_scaler.update(found_inf_flag)

        return found_inf_flag

    return False


@torch.no_grad()
def step_with_ready_grads(self) -> bool:
    """Step the optimizer with ready gradients, return successful."""
    timers = self.config.timers
    # Step the optimizer.
    if timers is not None:
        timers('optimizer-inner-step', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    self.optimizer.step()
    if timers is not None:
        timers('optimizer-inner-step').stop()

    # Update params from main params.
    if timers is not None:
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    if self.config.reuse_fp32_param:
        # fp32 -> bf16 + res
        self.fp32_tensor_convert_to_fp16_tensor()
    else:
        self._copy_main_params_to_model_params()
    if timers is not None:
        timers('optimizer-copy-main-to-model-params').stop()

    return True


@torch.no_grad()
def mixed_precision_optimizer_step(self):
    # Copy gradients from model params to main params.
    timers = self.config.timers
    timers('optimizer-copy-to-main-grad', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    self._copy_model_grads_to_main_grads()
    timers('optimizer-copy-to-main-grad').stop()
    if self.config.reuse_fp32_param:
        # bf16 -> fp32
        self.fp16_tensor_convert_to_fp32_tensor()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if self.grad_scaler:

        # Unscale and check for inf/nan.
        timers('optimizer-unscale-and-check-inf', log_level=1).start(
            barrier=self.config.barrier_with_L1_time)
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        timers('optimizer-unscale-and-check-inf').stop()

        # We are done with scaling gradients
        # so we can update the loss scale.
        self.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            return False, None, None

    # Clip the main gradients.
    timers('optimizer-clip-main-grad', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    grad_norm = None
    if self.config.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.config.clip_grad)
    timers('optimizer-clip-main-grad').stop()


    # Count the zeros in the grads.
    timers('optimizer-count-zeros', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    num_zeros_in_grad = self.count_zeros() if \
        self.config.log_num_zeros_in_grad else None
    timers('optimizer-count-zeros').stop()

    # Step the optimizer.
    timers('optimizer-inner-step', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    self.optimizer.step()
    timers('optimizer-inner-step').stop()

    # Update params from main params.
    timers('optimizer-copy-main-to-model-params', log_level=1).start(
        barrier=self.config.barrier_with_L1_time)
    if self.config.reuse_fp32_param:
        # fp32 -> bf16 + res
        self.fp32_tensor_convert_to_fp16_tensor()
    else:
        self._copy_main_params_to_model_params()
    timers('optimizer-copy-main-to-model-params').stop()

    # Successful update.
    return True, grad_norm, num_zeros_in_grad


def optimizer_config_init_wrapper(init_func):
    @wraps(init_func)
    def optimizer_config_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args = get_args()
        self.reuse_fp32_param = args.reuse_fp32_param if hasattr(args, "reuse_fp32_param") else False

    return optimizer_config_init


def get_megatron_optimizer_func_wrapper(func):
    @wraps(func)
    def get_megatron_optimizer_func(*args, **kwargs):
        chained_optimizer = func(*args, **kwargs)
        args = get_args()
        if hasattr(chained_optimizer, "chained_optimizers"):
            for optim in chained_optimizer.chained_optimizers:
                optim.optimizer.ema_decay = args.ema_decay
            return chained_optimizer
        if hasattr(chained_optimizer, "optimizer"):
            chained_optimizer.optimizer.ema_decay = args.ema_decay
            return chained_optimizer
        return chained_optimizer

    return get_megatron_optimizer_func


def reuse_fp32_param_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args = get_args()
        self.reuse_fp32_param = args.reuse_fp32_param if hasattr(args, "reuse_fp32_param") else False
        if self.reuse_fp32_param:
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
                                int32_float32_params_this_group[-1]
                        )
                self.res_float16_groups.append(res_float16_params_this_group)
                self.float16_float32_groups.append(float16_float32_params_this_group)
                self.int32_float32_groups.append(int32_float32_params_this_group)
            self._copy_model_params_to_main_params = _copy_model_params_to_main_params
            if args.npu_deterministic:
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
    """
    init a list of tensor with length of 2*fp32_tensor.numel() in bf16 to share the same storage.
    Args:
        fp32_tensor: original fp32 tensor.
        bf16_fp32_tensor: a bf16 tensor share the same storage with original list of fp32 tensors.
        res_tensor: a bf16 tensor that store the residual value of fp32 to bf16, shares a half of the
        storage with bf16_fp32_tensor.
        bf16_tensor: a bf16 tensor that store the value from fp32, shares another half of the
        storage with bf16_fp32_tensor.
        int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
    """
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
    """
    fp32(0p0p0p0p) -> bf16(pppp) + res(0000)
    rearrange the storage of bf16_fp32_tensors so that recover the fp32_tensors.
    Args:
        int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
        bf16_fp32_tensors: a list of bf16 tensors share the same storages with original list of fp32 tensors.
    Returns:
        None
    """
    for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
        if bf16_fp32_tensor.numel() == 0:
            return  
        int32_tensor.add_(32768)
        bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())


def bf16_tensors_to_fp32_tensors(int32_tensors, bf16_fp32_tensors):
    """
    res(0000) + bf16(pppp) -> fp32(0p0p0p0p)
    rearrange the storage of bf16_fp32_tensors so that recover the fp32_tensors.
    Args:
        int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
        bf16_fp32_tensors: a list of bf16 tensors share the same storages with original list of fp32 tensors.
    Returns:
        None
    """
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
        if hasattr(optimizer.state[fp32_param]["exp_avg_sq"], "meta"):
            exp_avg_sq = optimizer.state[fp32_param]["exp_avg_sq"].meta.dequantization(
                optimizer.state[fp32_param]["exp_avg_sq"].data)
            exp_avg_sq.mul_(sign_tensor)
            optimizer.state[fp32_param]["exp_avg_sq"].data.copy_(
                optimizer.state[fp32_param]["exp_avg_sq"].meta.quantization(exp_avg_sq.data))
        else:
            optimizer.state[fp32_param]["exp_avg_sq"].mul_(sign_tensor)


def optimizer_exp_avg_load_sign(optimizer, fp32_param, int32_tensor):
    if "exp_avg_sq" in optimizer.state[fp32_param]:
        if hasattr(optimizer.state[fp32_param]["exp_avg_sq"], "meta"):
            exp_avg_sq = optimizer.state[fp32_param]["exp_avg_sq"].meta.dequantization(
                optimizer.state[fp32_param]["exp_avg_sq"].data)
            odd_even_tensor = (torch.sign(exp_avg_sq) > 0).reshape(-1)
        else:
            odd_even_tensor = (torch.sign(optimizer.state[fp32_param]["exp_avg_sq"]) > 0).reshape(-1)
        optimizer.state[fp32_param]["exp_avg_sq"].abs_()
        int32_tensor.add_(odd_even_tensor)


def _collect_main_grad_data_for_unscaling_wrapper(func):
    @wraps(func)
    def _collect_main_grad_data_for_unscaling_func(self):
        main_grads = func(self)
        meta_grads_scale_inv = []
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if hasattr(main_param, "quant_grad"):
                    meta_grads_scale_inv.append(main_param.quant_grad.meta.scale_inv)

        return main_grads, meta_grads_scale_inv

    return _collect_main_grad_data_for_unscaling_func


def _copy_model_grads_to_main_grads(self):
    # This only needs to be done for the float16 group.
    args = get_args()
    for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
        for model_param, main_param in zip(model_group, main_group):
            if hasattr(model_param, 'main_grad'):
                if args.quant_grads:
                    main_param.quant_grad = model_param.main_grad
                else:
                    main_param.grad = model_param.main_grad.float()
            else:
                if model_param.grad is not None:
                    if args.quant_grads:
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


def _unscale_main_grads_and_check_for_nan(self):

    # Collect main grads.
    main_grads, meta_grads_scale_inv = self._collect_main_grad_data_for_unscaling()

    # Reset found inf.
    self.found_inf.fill_(0.0)

    # Unscale and set found inf/nan
    torch._amp_foreach_non_finite_check_and_unscale_(
        main_grads, self.found_inf, self.grad_scaler.inv_scale
    )

    if len(meta_grads_scale_inv) > 0:
        torch._amp_foreach_non_finite_check_and_unscale_(
            meta_grads_scale_inv, self.found_inf, self.grad_scaler.inv_scale
        )

    # Update across all model parallel instances.
    torch.distributed.all_reduce(
        self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()
    )

    # Check for nan.
    found_inf_flag = self.found_inf.item() > 0

    return found_inf_flag


def get_main_grads_for_grad_norm(self):
    """
    Get main_grads that should be taken into account to compute the grad norm.
    Filter parameters based on:
      - grad should not be None.
      - parameter should not be shared (i.e., grads shouldn't be double counted while
        computing norms).
      - should not be a replica due to tensor model parallelism.
    """
    params = self.get_parameters()
    grads_for_norm = []
    for param in params:
        if hasattr(param, "quant_grad"):
            grad = param.quant_grad
        else:
            grad = param.grad
        grad_not_none = grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    return grads_for_norm


def _zero_grad_group_helper_wrapper(func):
    @wraps(func)
    def _zero_grad_group_helper_func(group: List[torch.nn.Parameter], set_to_none: bool):
        func(group, set_to_none)
        for param in group:
            if hasattr(param, "quant_grad"):
                if set_to_none:
                    param.quant_grad = None
                else:
                    if param.quant_grad.grad_fn is not None:
                        param.quant_grad.detach_()
                    else:
                        param.quant_grad.requires_grad_(False)
                    param.quant_grad.zero_()
    return _zero_grad_group_helper_func