# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import torch
import torch_npu
from einops import rearrange
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32


class QuantDtype:
    def __init__(self, x: torch.dtype, w: torch.dtype, grads: torch.dtype):
        self.x = x
        self.w = w
        self.grads = grads


def get_quant_dtype(quant_format):
    if quant_format == "E4M3":
        return QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn)

    elif quant_format == "HIF8":
        return QuantDtype(torch_npu.hifloat8, torch_npu.hifloat8, torch_npu.hifloat8)

    elif quant_format == "HYBRID":
        return QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)

    else:
        raise ValueError("Unknown quant format")


class BaseGMMFunction(torch.autograd.Function):
    @classmethod
    def gmm_apply(cls, x, weight, bias, tokens_per_expert, weight_param, quant_format,
                  gemm_gradient_accumulation_fusion):
        if isinstance(tokens_per_expert, list):
            tokens_per_expert = torch.tensor(tokens_per_expert, device='npu', dtype=torch.int64)
        group_list = torch.cumsum(tokens_per_expert, dim=0)
        return cls.apply(x, weight, bias, group_list, weight_param, 0, quant_format, gemm_gradient_accumulation_fusion)

    @classmethod
    def forward(cls, ctx, x, weight, bias, group_list, weight_param, group_list_type=0, quant_format='E4M3',
                gemm_gradient_accumulation_fusion=False):
        if isinstance(group_list, torch.Tensor):
            if group_list.device.type == 'cpu':
                group_list = group_list.npu()
        else:
            group_list = torch.tensor(group_list, device='npu', dtype=torch.int64)
        output = cls.op_forward(x, weight, group_list, group_list_type, bias=bias, quant_format=quant_format)
        ctx.save_for_backward(x, weight, group_list)
        ctx.weight_param = weight_param
        ctx.group_list_type = group_list_type
        ctx.quant_format = quant_format
        ctx.gemm_gradient_accumulation_fusion = gemm_gradient_accumulation_fusion
        return output[0]

    @classmethod
    def backward(cls, ctx, grad_outputs):
        x, weight, group_list = ctx.saved_tensors
        weight_param = ctx.weight_param
        group_list_type = ctx.group_list_type
        quant_format = ctx.quant_format
        dx = cls.op_dx(grad_outputs, weight, group_list, group_list_type, quant_format=quant_format)[0]

        if ctx.gemm_gradient_accumulation_fusion:
            grad_weights = cls.op_gmm_add(x, weight, grad_outputs, group_list, weight_param, quant_format=quant_format)
        else:
            grad_weights = cls.op_dw(x, grad_outputs, group_list, group_list_type, quant_format=quant_format)[0]
        return dx, grad_weights, None, None, None, None, None, None

    @classmethod
    def op_forward(cls, x, weight, group_list, group_list_type=0, bias=None, quant_format='E4M3'):
        # x * weight
        raise NotImplementedError

    @classmethod
    def op_dx(cls, grad, weight, group_list, group_list_type=0, bias=None, quant_format='E4M3'):
        # grad * wt
        raise NotImplementedError

    @classmethod
    def op_dw(cls, x, grad, group_list, group_list_type=0, bias=None, quant_format='E4M3'):
        # xt * grad
        raise NotImplementedError

    @classmethod
    def op_gmm_add(cls, x, weight, grad, group_list, weight_param, quant_format):
        cls.gmm_add_impl(x, grad, group_list, weight_param, weight.shape, quant_format)
        grad_weights = torch.zeros(
            weight.shape,
            dtype=x.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        return grad_weights

    @classmethod
    def gmm_add_impl(cls, x, grad, group_list, weight_param, weight_shape, quant_format):
        npu_groupmatmul_add_fp32(x, grad, group_list, weight_param.grad)


class MXFP8GMMFunction(BaseGMMFunction):
    @classmethod
    def op_forward(cls, x, weight, group_list, group_list_type=0, bias=None, quant_format='E4M3'):
        qdtype = get_quant_dtype(quant_format)
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=qdtype.x)
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-2, dst_type=qdtype.w)
        return torch_npu.npu_grouped_matmul([x_mxfp8], [weight_mxfp8], bias=bias,
                                            scale=[weight_scale], per_token_scale=[x_scale],
                                            group_list=group_list, group_type=0,
                                            output_dtype=x.dtype, group_list_type=group_list_type,
                                            scale_dtype=torch_npu.float8_e8m0fnu,
                                            per_token_scale_dtype=torch_npu.float8_e8m0fnu, split_item=3)

    @classmethod
    def op_dx(cls, grad, weight, group_list, group_list_type=0, bias=None, quant_format='E4M3'):
        qdtype = get_quant_dtype(quant_format)
        grad_mxfp8, grad_scale = torch_npu.npu_dynamic_mx_quant(grad, axis=-1, dst_type=qdtype.grads)
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-1, dst_type=qdtype.w)
        return torch_npu.npu_grouped_matmul([grad_mxfp8], [rearrange(weight_mxfp8, 'n h f -> n f h')], bias=bias,
                                            scale=[rearrange(weight_scale, 'n h f g -> n f h g')],
                                            per_token_scale=[grad_scale], group_list=group_list, group_type=0,
                                            output_dtype=grad.dtype, group_list_type=group_list_type,
                                            scale_dtype=torch_npu.float8_e8m0fnu,
                                            per_token_scale_dtype=torch_npu.float8_e8m0fnu, split_item=3)

    @classmethod
    def op_dw(cls, x, grad, group_list, group_list_type=0, bias=None, quant_format='E4M3'):
        qdtype = get_quant_dtype(quant_format)
        x_mxfp8, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            x, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.x, blocksize=32)
        grad_mxfp8, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.grads, blocksize=32)
        return torch_npu.npu_grouped_matmul([x_mxfp8.t()], [grad_mxfp8], bias=bias, scale=[grad_scale],
                                            per_token_scale=[rearrange(x_scale, 'n h f -> h n f')],
                                            group_list=group_list, group_type=2, output_dtype=x.dtype,
                                            group_list_type=group_list_type,
                                            scale_dtype=torch_npu.float8_e8m0fnu,
                                            per_token_scale_dtype=torch_npu.float8_e8m0fnu, split_item=3)

    @classmethod
    def gmm_add_impl(cls, x, grad, group_list, weight_param, weight_shape, quant_format='E4M3'):
        qdtype = get_quant_dtype(quant_format)
        x_quant, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            x, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.x, blocksize=32)
        grad_quant, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.grads, blocksize=32)
        if weight_param.grad is not None:
            torch_npu.npu_add_quant_gmm_(weight_param.grad.view(weight_shape),
                                         x_quant.t(), grad_quant, grad_scale,
                                         x1_scale=rearrange(x_scale, 'n h f -> h n f'),
                                         group_list_type=0, group_list=group_list,
                                         x1_scale_dtype=torch_npu.float8_e8m0fnu,
                                         x2_scale_dtype=torch_npu.float8_e8m0fnu)


def npu_quant_group_gemm(x, weight, bias, tokens_per_expert, weight_param, quant_format, gemm_gradient_accumulation_fusion):
    return MXFP8GMMFunction.gmm_apply(
        x,
        weight,
        bias,
        tokens_per_expert,
        weight_param,
        quant_format,
        gemm_gradient_accumulation_fusion

    )
