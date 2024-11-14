import torch
from torch.library import impl
from mindspeed.op_builder import GMMOpBuilder, GMMV2OpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


__all__ = ["npu_gmm", "npu_gmm_v2"]


def check_optional_tensor(tensor, device, name):
    if not isinstance(tensor, (torch.Tensor, type(None))):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)}.")
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but found at least two devices, "
            f"{device}(arg0) and {tensor.device}({name})!")


class GMMFunction(torch.autograd.Function):
    builder = GMMOpBuilder()
    builder2 = GMMV2OpBuilder()

    @staticmethod
    def forward(ctx, original_weight, x, weight, bias, group_args):
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        bias = [] if bias is None else [bias]
        if group_list_type == 0:
            outputs = GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
        elif group_list_type == 1:
            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight, original_weight)
            ctx.group_list = group_list
        else:
            ctx.save_for_backward(x, weight, group_list, original_weight)
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        ctx.group_list_data_type = group_list_data_type


        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        if ctx.group_list_data_type == 0:
            x, weight, original_weight = ctx.saved_tensors
            group_list = ctx.group_list
        else:
            x, weight, group_list, original_weight = ctx.saved_tensors

        if ctx.gemm_fusion:
            if ctx.group_list_type == 0:
                dx, _, dbias = GMMFunction.builder.load().npu_gmm_backward_fusion([grad_outputs], [weight], group_list,
                                                                    ctx.group_list_type)
                npu_groupmatmul_add_fp32(x, grad_outputs, group_list, original_weight.main_grad)
                
            elif ctx.group_list_type == 1:
                dx, _, dbias = GMMFunction.builder2.load().npu_gmm_backward_fusion([grad_outputs], [weight], group_list,
                                                                    ctx.group_list_type)
                group_list_v2 = torch.cumsum(group_list, dim=0)                                           
                npu_groupmatmul_add_fp32(x, grad_outputs, group_list_v2, original_weight.main_grad)

            dbias = None if len(dbias) == 0 else dbias[0]
  
            if hasattr(original_weight, 'grad_added_to_main_grad'):
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.shape,
                        dtype=x.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.shape,
                        dtype=x.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                original_weight.grad_added_to_main_grad = True
            else:
                grad_weight = None

            return None, dx[0], grad_weight, dbias, None
        else:
            if ctx.group_list_type == 0:
                dx, dw, dbias = GMMFunction.builder.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
                                                                    ctx.group_list_type)
            elif ctx.group_list_type == 1:
                dx, dw, dbias = GMMFunction.builder2.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
                                                                    ctx.group_list_type)
            dbias = None if len(dbias) == 0 else dbias[0]

            return None, dx[0], dw[0], dbias, None



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
    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list, group_type=group_type,
                               group_list_type=group_list_type)
    if group_list_type == 0:
        return torch.ops.mindspeed.npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
    elif group_list_type == 1:
        return torch.ops.mindspeed.npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
    else:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")


@impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
@impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
def _npu_gmm(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    if isinstance(group_list, (torch.Tensor, type(None))):
        group_list_data_type = 1
    else:
        group_list_data_type = 0
    group_args = (group_list, group_type, gemm_fusion, 0, group_list_data_type)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)


@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
def _npu_gmm_v2(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
    group_args = (group_list, group_type, gemm_fusion, 1, 1)
    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=1, gemm_fusion=gemm_fusion)
