# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Type

import torch
import torch_npu
from einops import rearrange

from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.weight_grad_store import WeightGradStore
from mindspeed.ops.npu_groupmatmul_add import npu_groupmatmul_add_fp32
from mindspeed.ops.npu_matmul_add import is_a5
from mindspeed.te.pytorch.fp8.constants import Fp8Recipe, TensorKey
from mindspeed.te.pytorch.fp8.reuse import reuse_or_quantize
from mindspeed.te.pytorch.utils import get_quant_dtype


class GmmContext:
    pass


def get_gmm_quant_func() -> Type['BaseGMMFunction'] | None:
    args = get_args()
    if not args.fp8 or not args.use_gmm_fp8:
        return None
    if args.fp8_recipe == Fp8Recipe.mxfp8:
        return MXFP8GMMFunction
    elif args.fp8_recipe == Fp8Recipe.mxfp8_32x32:
        return MXFP832x32GMMFunction
    elif args.fp8_recipe in [Fp8Recipe.tensorwise, Fp8Recipe.delayed]:
        return TensorwiseGMMFunction
    # Blockwise FP8 is not implemented here yet, so fall back to the high-precision path.
    return None


def get_gmm_op_cls() -> Type['BaseGMMFunction']:
    gmm_quant_func = get_gmm_quant_func()
    if gmm_quant_func is not None:
        return gmm_quant_func
    return BF16GMMFunction


class BaseGMMFunction(torch.autograd.Function):
    @classmethod
    def gmm_apply(cls, x, weight, bias, tokens_per_expert, weight_param):
        # Accept tokens_per_expert and normalize it into group_list.
        if isinstance(tokens_per_expert, list):
            tokens_per_expert = torch.tensor(tokens_per_expert, device='npu', dtype=torch.int64)
        group_list = torch.cumsum(tokens_per_expert, dim=0)
        return cls.apply(x, weight, bias, group_list, weight_param)

    @classmethod
    def forward(cls, ctx, x, weight, bias, group_list, weight_param, group_list_type=0):
        if isinstance(group_list, torch.Tensor):
            if group_list.device.type == 'cpu':
                group_list = group_list.npu()
        else:
            group_list = torch.tensor(group_list, device='npu', dtype=torch.int64)
        output = cls.op_forward(ctx, x, weight, group_list, group_list_type, bias=bias)
        ctx.save_for_backward(x, weight, group_list)
        ctx.weight_param = weight_param
        ctx.group_list_type = group_list_type
        return output[0]

    @classmethod
    def backward(cls, ctx, grad_outputs):
        x, weight, group_list = ctx.saved_tensors
        weight_param = ctx.weight_param
        group_list_type = ctx.group_list_type
        dx = cls.op_dx(ctx, grad_outputs, weight, group_list, group_list_type)[0]
        if WeightGradStore.is_decoupleBlock:
            # Split dw computation and defer it to the delayed path.
            weight_tensor = rearrange(weight, 'n h f -> h n f')
            WeightGradStore.put(
                [ctx, x, group_list, group_list_type, weight.shape],
                grad_outputs,
                weight_param,
                sequence_parallel=False,
                in_row=False,
            )
            if hasattr(weight_param, 'grad_added_to_main_grad') and getattr(get_args(), 'overlap_grad_reduce', False):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                shape = list(weight_tensor.shape)
                shape[1], shape[2] = shape[2], shape[1]
                weight_param.skip_grad_accum = True

            grad_weights = None
        elif get_args().gemm_gradient_accumulation_fusion:
            grad_weights = cls.op_gmm_add(x, weight, grad_outputs, group_list, weight_param)
        else:
            grad_weights = cls.op_dw(ctx, x, grad_outputs, group_list, group_list_type)[0]
        return dx, grad_weights, None, None, None, None

    @classmethod
    def op_forward(cls, ctx, x, weight, group_list, group_list_type=0, bias=None):
        # x * weight
        raise NotImplementedError

    @classmethod
    def op_dx(cls, ctx, grad, weight, group_list, group_list_type=0, bias=None):
        # grad * wt
        raise NotImplementedError

    @classmethod
    def op_dw(cls, ctx, x, grad, group_list, group_list_type=0, bias=None):
        # xt * grad
        raise NotImplementedError

    @classmethod
    def op_gmm_add(cls, x, weight, grad, group_list, weight_param):
        cls.gmm_add_impl(x, grad, group_list, weight_param, weight.shape)
        if hasattr(weight_param, 'grad_added_to_main_grad'):
            if getattr(weight, 'zero_out_wgrad', False):
                grad_weights = torch.zeros(
                    weight.shape,
                    dtype=x.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                grad_weights = torch.empty(
                    weight.shape,
                    dtype=x.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            weight_param.grad_added_to_main_grad = True
        else:
            grad_weights = None
        return grad_weights

    @classmethod
    def gmm_add_impl(cls, x, grad, group_list, weight_param, weight_shape):
        npu_groupmatmul_add_fp32(x, grad, group_list, weight_param.main_grad)


class BF16GMMFunction(BaseGMMFunction):
    @classmethod
    def op_forward(cls, ctx, x, weight, group_list, group_list_type=0, bias=None):
        if not is_a5():
            from mindspeed.ops.gmm import GMMFunction

            return GMMFunction.builder.load().npu_gmm([x], [weight], bias or [], group_list, 0, 0)

        return torch_npu.npu_grouped_matmul(
            [x], [weight], bias=bias, group_list=group_list, split_item=3, group_type=0, group_list_type=group_list_type
        )

    @classmethod
    def op_dx(cls, ctx, grad, weight, group_list, group_list_type=0, bias=None):
        if len(weight.shape) == 3:
            weight = rearrange(weight, 'n h f -> n f h')
        else:
            weight = weight.t()
        if not is_a5():
            from mindspeed.ops.gmm import GMMFunction

            return GMMFunction.builder.load().npu_gmm([grad], [weight], bias or [], group_list, 0, 0)
        return torch_npu.npu_grouped_matmul(
            [grad],
            [weight],
            bias=bias,
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
        )

    @classmethod
    def op_dw(cls, ctx, x, grad, group_list, group_list_type=0, bias=None):
        if not is_a5():
            from mindspeed.ops.gmm import GMMFunction

            return GMMFunction.builder.load().npu_gmm([x.t()], [grad], bias or [], group_list, 2, 0)
        return torch_npu.npu_grouped_matmul(
            [x.t()],
            [grad],
            bias=bias,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )


class MXFP8GMMFunction(BaseGMMFunction):
    @classmethod
    def op_forward(cls, ctx, x, weight, group_list, group_list_type=0, bias=None, reuse_identity=None):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=qdtype.x)
        weight_col_mxfp8, weight_col_scale, weight_row_mxfp8, weight_row_scale = reuse_or_quantize(
            weight,
            TensorKey.weight,
            torch_npu.npu_dynamic_mx_quant_with_dual_axis,
            op_name="npu_dynamic_mx_quant_with_dual_axis",
            reuse_identity=reuse_identity,
            dst_type=qdtype.w,
        )
        ctx.w_quant = (weight_col_mxfp8, weight_col_scale)
        return torch_npu.npu_grouped_matmul(
            [x_mxfp8],
            [weight_row_mxfp8],
            bias=bias,
            scale=[weight_row_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )

    @classmethod
    def op_dx(cls, ctx, grad, weight, group_list, group_list_type=0, bias=None):
        qdtype = get_quant_dtype()
        grad_mxfp8, grad_scale = torch_npu.npu_dynamic_mx_quant(grad, axis=-1, dst_type=qdtype.grads)
        weight_mxfp8, weight_scale = ctx.w_quant
        return torch_npu.npu_grouped_matmul(
            [grad_mxfp8],
            [rearrange(weight_mxfp8, 'n h f -> n f h')],
            bias=bias,
            scale=[rearrange(weight_scale, 'n h f g -> n f h g')],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=grad.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )

    @classmethod
    def op_dw(cls, ctx, x, grad, group_list, group_list_type=0, bias=None):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            x, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.x, blocksize=32
        )
        grad_mxfp8, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.grads, blocksize=32
        )
        return torch_npu.npu_grouped_matmul(
            [x_mxfp8.t()],
            [grad_mxfp8],
            bias=bias,
            scale=[grad_scale],
            per_token_scale=[rearrange(x_scale, 'n h f -> h n f')],
            group_list=group_list,
            group_type=2,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )

    @classmethod
    def gmm_add_impl(cls, x, grad, group_list, weight_param, weight_shape):
        qdtype = get_quant_dtype()
        x_quant, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            x, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.x, blocksize=32
        )
        grad_quant, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad, group_list.to(torch.int32), round_mode="rint", dst_type=qdtype.grads, blocksize=32
        )
        torch_npu.npu_add_quant_gmm_(
            weight_param.main_grad.view(weight_shape),
            x_quant.t(),
            grad_quant,
            grad_scale,
            x1_scale=rearrange(x_scale, 'n h f -> h n f'),
            group_list_type=0,
            group_list=group_list,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )


class TensorwiseGMMFunction(BaseGMMFunction):
    @classmethod
    def op_forward(cls, ctx, x, weight, group_list, group_list_type=0, bias=None):
        qdtype = get_quant_dtype()
        g_size = len(group_list)
        x_quant, x_scale = torch_npu.npu_dynamic_quant(
            x.view(g_size, -1),
            dst_type=qdtype.x,
            quant_mode='pertensor',
        )
        x_scale = x_scale.expand(len(group_list))
        w_quant, w_scale = reuse_or_quantize(
            weight.view(g_size, -1),
            TensorKey.weight,
            torch_npu.npu_dynamic_quant,
            dst_type=qdtype.w,
        )
        x_quant = x_quant.view(x.shape)
        w_quant = w_quant.view(weight.shape)
        ctx.saved_x = (x_quant, x_scale)
        ctx.saved_weight = (w_quant, w_scale)
        ctx.g_size = g_size
        return torch_npu.npu_grouped_matmul(
            [x_quant],
            [w_quant],
            scale=[w_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            bias=bias,
            split_item=3,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            **qdtype.gmm_kwargs,
        )

    @classmethod
    def op_dx(cls, ctx, grad, weight, group_list, group_list_type=0, bias=None):
        qdtype = get_quant_dtype()
        grad_quant, grad_scale = torch_npu.npu_dynamic_quant(
            grad.view(ctx.g_size, -1),
            dst_type=qdtype.grads,
            quant_mode='pertensor',
        )
        grad_scale = grad_scale.expand(len(group_list))
        grad_quant = grad_quant.view(grad.shape)

        w_quant, w_scale = ctx.saved_weight
        ctx.saved_grads = (grad_quant, grad_scale)
        return torch_npu.npu_grouped_matmul(
            [grad_quant],
            [rearrange(w_quant, 'n h f -> n f h')],
            bias=None,
            scale=[w_scale],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            split_item=3,
            output_dtype=grad.dtype,
            group_list_type=0,
            **qdtype.gmm_kwargs,
        )

    @classmethod
    def op_dw(cls, ctx, x, grad, group_list, group_list_type=0, bias=None):
        qdtype = get_quant_dtype()
        x_quant, x_scale = ctx.saved_x
        grad_quant, grad_scale = ctx.saved_grads
        return torch_npu.npu_grouped_matmul(
            [x_quant.t()],
            [grad_quant],
            scale=[grad_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=2,
            bias=bias,
            split_item=3,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            **qdtype.gmm_kwargs,
        )


class MXFP832x32GMMFunction(MXFP8GMMFunction):
    @classmethod
    def op_forward(cls, ctx, x, weight, group_list, group_list_type=0, bias=None, reuse_identity=None):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, axis=-1, dst_type=qdtype.x)
        weight_col_mxfp8, weight_col_scale, weight_row_scale = reuse_or_quantize(
            weight,
            TensorKey.weight,
            torch_npu.npu_dynamic_block_mx_quant,
            op_name="npu_dynamic_block_mx_quant",
            reuse_identity=reuse_identity,
            dst_type=qdtype.w,
        )
        weight_row_mxfp8 = weight_col_mxfp8
        ctx.w_quant = (weight_col_mxfp8, weight_col_scale)
        return torch_npu.npu_grouped_matmul(
            [x_mxfp8],
            [weight_row_mxfp8],
            bias=bias,
            scale=[weight_row_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )
