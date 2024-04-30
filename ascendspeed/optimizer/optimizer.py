import os
from functools import wraps

import torch
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params


@torch.no_grad()
def mixed_precision_optimizer_step(self, args, timers):
    # Copy gradients from model params to main params.
    timers('optimizer-copy-to-main-grad', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    self._copy_model_grads_to_main_grads()
    timers('optimizer-copy-to-main-grad').stop()
    if args.reuse_fp32_param:
        # bf16 -> fp32
        for int32_float32_group, float16_param_group in zip(
            self.int32_float32_groups, self.float16_float32_groups):
            bf16_tensors_to_fp32_tensors(int32_float32_group, float16_param_group)

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if self.grad_scaler:

        # Unscale and check for inf/nan.
        timers('optimizer-unscale-and-check-inf', log_level=1).start(
            barrier=args.barrier_with_L1_time)
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
        barrier=args.barrier_with_L1_time)
    grad_norm = None
    if self.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.clip_grad,
                                        self.check_for_nan_in_grad)
    timers('optimizer-clip-main-grad').stop()


    # Count the zeros in the grads.
    timers('optimizer-count-zeros', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    num_zeros_in_grad = self.count_zeros() if \
        self.log_num_zeros_in_grad else None
    timers('optimizer-count-zeros').stop()

    # Step the optimizer.
    timers('optimizer-inner-step', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    self.optimizer.step()
    timers('optimizer-inner-step').stop()

    # Update params from main params.
    timers('optimizer-copy-main-to-model-params', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    if args.reuse_fp32_param:
        # fp32 -> bf16 + res
        for int32_float32_param_group, float16_param_group in zip(
            self.int32_float32_groups, self.float16_float32_groups):
            fp32_tensors_to_bf16_tensors(int32_float32_param_group, float16_param_group)
    else:
        self._copy_main_params_to_model_params()
    timers('optimizer-copy-main-to-model-params').stop()

    # Successful update.
    return True, grad_norm, num_zeros_in_grad


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
    return reuse_fp32_param_init


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
    from ascendspeed.op_builder import AlgorithmOpBuilder
    reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
    reuse_data_ptr(bf16_fp32_tensor, fp32_tensor, 0)
    reuse_data_ptr(int32_tensor, fp32_tensor, 0)
    fp32_tensors_to_bf16_tensors([int32_tensor], [bf16_fp32_tensor])
    reuse_data_ptr(res_tensor, bf16_fp32_tensor, 0)
    reuse_data_ptr(bf16_tensor, bf16_fp32_tensor, res_tensor.numel())


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
