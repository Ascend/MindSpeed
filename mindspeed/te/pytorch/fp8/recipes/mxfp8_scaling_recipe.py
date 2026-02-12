import dataclasses

import torch

import torch_npu
from mindspeed.te.pytorch.fp8.tensor import MXFP8Tensor
from mindspeed.te.pytorch.fp8.constants import TensorKey
from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling
from mindspeed.te.pytorch.utils import view_as_n_dim, get_quant_dtype


class MXFP8ScalingRecipe(Recipe):
    need_transpose_key = (TensorKey.weight, TensorKey.grads)

    def quantization(self, tensor: torch.Tensor, key, colwise, rowwise):
        if tensor is None:
            return tensor
        coly, col_scale, rowy, row_scale = None, None, None, None
        tensor_2d = view_as_n_dim(tensor)
        fp8_dtype = self.quant_dtype
        mxfp8_tensor = MXFP8Tensor(fp8_dtype, tensor.shape, tensor.dtype)

        if rowwise and colwise:
            coly, col_scale, rowy, row_scale = \
                torch_npu.npu_dynamic_mx_quant_with_dual_axis(tensor_2d, dst_type=fp8_dtype)
        elif colwise:
            coly, col_scale = torch_npu.npu_dynamic_mx_quant(tensor_2d, axis=-1, dst_type=fp8_dtype)
        elif rowwise:
            rowy, row_scale = torch_npu.npu_dynamic_mx_quant(tensor_2d, axis=-2, dst_type=fp8_dtype)

        # forward: x.col   @ w.col.T
        # dx     : g.col   @ w.row
        # dw     : g.row.T @ x.row
        mxfp8_tensor.set_row_data(rowy, row_scale, key == TensorKey.grads)
        mxfp8_tensor.set_col_data(coly, col_scale, key == TensorKey.weight)

        return mxfp8_tensor


@dataclasses.dataclass
class MXFP8BlockScaling(RecipeScaling):
    recipe = MXFP8ScalingRecipe


class MXFP8MatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, need_grad: bool = True):
        qdtype = get_quant_dtype()
        x_2d = view_as_n_dim(x)
        ctx.output_dtype = x.dtype
        if need_grad:
            x_quant, x_scale, ctx.x, ctx.x_scale = \
                torch_npu.npu_dynamic_mx_quant_with_dual_axis(x_2d, dst_type=qdtype.x)
            w_quant, w_scale, ctx.w, ctx.w_scale = \
                torch_npu.npu_dynamic_mx_quant_with_dual_axis(weight, dst_type=qdtype.x)
        else:
            x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(x_2d, axis=-1, dst_type=qdtype.x)
            w_quant, w_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-1, dst_type=qdtype.w)
            ctx.save_for_backward(x, weight)
        output = torch_npu.npu_quant_matmul(x_quant, w_quant.t(), w_scale.transpose(0, 1),
                                            pertoken_scale=x_scale,
                                            output_dtype=x.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        if len(x.shape) != 2:
            output = output.reshape(*x.shape[:-1], *output.shape[1:])
        if weight.requires_grad:
            output.requires_grad = True
        return output

    @staticmethod
    def backward(ctx, grads: torch.Tensor):
        qdtype = get_quant_dtype()
        grads_dx, grads_dx_scale, grads_dw, grads_dw_scale = \
            torch_npu.npu_dynamic_mx_quant_with_dual_axis(view_as_n_dim(grads), dst_type=qdtype.grads)

        if hasattr(ctx, 'x'):
            x_quant, x_scale, w_quant, w_scale = ctx.x, ctx.x_scale, ctx.w, ctx.w_scale
        else:
            x, weight = ctx.saved_tensors
            w_quant, w_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-2, dst_type=qdtype.w)
            x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(x), axis=-2, dst_type=qdtype.x)

        dx = torch_npu.npu_quant_matmul(grads_dx, w_quant, w_scale,
                                        pertoken_scale=grads_dx_scale,
                                        output_dtype=ctx.output_dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        dw = torch_npu.npu_quant_matmul(grads_dw.t(), x_quant, x_scale, pertoken_scale=grads_dw_scale.transpose(0, 1),
                                        output_dtype=ctx.output_dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        return dx, dw, None, None, None
