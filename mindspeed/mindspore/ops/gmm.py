# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from torch.library import impl
from mindspore import ops
from mindspeed.op_builder import GMMOpBuilder, GMMV2OpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.ops.gmm import GMMFunction


__all__ = ["npu_gmm", "npu_gmm_v2"]


def check_optional_tensor(tensor, device, name):
    if not isinstance(tensor, (torch.Tensor, type(None))):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)}.")
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at least two devices, "
            f"{device}(arg0) and {tensor.device}({name})!")


def npu_gmm_param_verification(x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"arg0 must be a torch.Tensor, got {type(x)}.")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"arg1 must be a torch.Tensor, got {type(weight)}.")
    if not isinstance(bias, (torch.Tensor, type(None))):
        raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias)}.")
    if (group_list_type == 0):
        if not (
            isinstance(group_list, (torch.Tensor, type(None)))
            or (isinstance(group_list, list) and all(isinstance(x, int) for x in group_list))
        ):
            raise TypeError(f"group_list must be a List of int64, torch.Tensor or None, got {type(group_list)}.")
    else:
        if not (isinstance(group_list, (torch.Tensor, type(None)))):
            raise TypeError(f"group_list must be a torch.Tensor or None, got {type(group_list)}.")
    if isinstance(group_list, torch.Tensor):
        if len(group_list.shape) > 1:
            raise ValueError(f"If group_list is not None, it must be an one-dimensional tensor, "
                             f"got dimension of group_list: {len(group_list.shape)}!")
        if group_list.dtype != torch.int64:
            raise TypeError(f"group_list must be a List of int64, got group_list type: {type(group_list)}, "
                            f"dtype: {group_list.dtype}!")
    if not isinstance(group_type, (int, type(None))):
        raise TypeError(f"group_type must be an int or None, got {type(group_type)}.")
    # Ensure all tensors on the same device
    x_device = x.device
    device_warning = "Expected all tensors to be on the same device, but found at least two devices"
    if weight.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {weight.device}(arg1)!")
    if bias is not None and bias.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {bias.device}(bias)!")
    if isinstance(group_list, torch.Tensor) and group_list.device != x_device:
        raise RuntimeError(f"{device_warning}, {x_device}(arg0) and {group_list.device}(group_list)!")


def _npu_gmm_common(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0, gemm_fusion=False):
    support_dtype = [torch.float16, torch.bfloat16, torch.float32]
    if weight.dtype not in support_dtype:
        raise TypeError(f"Only support non quant case, but got weight dtype {weight.dtype}.")
    if group_list_type == 0:
        return _npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
    elif group_list_type == 1:
        return _npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
    else:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")


# @impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
# @impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
def _npu_gmm(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    group_args = (group_list, group_type, gemm_fusion, 0, 0)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list.tolist(), group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)


def _npu_gmm_v2(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    group_args = (group_list, group_type, gemm_fusion, 1, 0)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=1, gemm_fusion=gemm_fusion)


class _GmmProxy:
    def npu_gmm(self, *args, **kwargs):
        return ops.function.math_func.gmm(*args, **kwargs)
    
    def npu_gmm_backward(self, *args, **kwargs):
        return ops.function.math_func.gmm_backward(*args, **kwargs)
    
    def npu_gmm_backward_fusion(self, grad_outputs, weight, group_list, group_list_type):
        return ops.function.math_func.gmm_backward(grad_outputs, [torch.ones_like(g) for g in grad_outputs], weight, group_list, group_list_type)


class _GmmProxy2:
    def npu_gmm(self, *args, **kwargs):
        return ops.function.math_func.gmm(*args, **kwargs)
    
    def npu_gmm_backward(self, *args, **kwargs):
        return ops.function.math_func.gmm_v2_backward(*args, **kwargs)
    
    def npu_gmm_backward_fusion(self, grad_outputs, weight, group_list, group_list_type):
        return ops.function.math_func.gmm_v2_backward(grad_outputs, [torch.ones_like(g) for g in grad_outputs], weight, group_list, group_list_type)
    

_GMM_PROXY = _GmmProxy()
_GMM_PROXY2 = _GmmProxy2()


def _GMM_patched_load(*_args, **_kwargs):
    return _GMM_PROXY


def _GMM_patched_load2(*_args, **_kwargs):
    return _GMM_PROXY2