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

        mxfp8_tensor = MXFP8Tensor(self.fp8_format_dtype, tensor.shape, tensor.dtype)
        if colwise:
            coly, col_scale = torch_npu.npu_dynamic_mx_quant(tensor_2d, axis=-1, dst_type=self.fp8_format_dtype)
        if rowwise:
            rowy, row_scale = torch_npu.npu_dynamic_mx_quant(tensor_2d, axis=-2, dst_type=self.fp8_format_dtype)

        if key == TensorKey.weight:
            coly, col_scale, rowy, row_scale = rowy, row_scale, coly, col_scale

        mxfp8_tensor.set_row_data(rowy, row_scale, key in self.need_transpose_key)
        mxfp8_tensor.set_col_data(coly, col_scale)

        return mxfp8_tensor


@dataclasses.dataclass
class MXFP8BlockScaling(RecipeScaling):
    recipe = MXFP8ScalingRecipe


class MXFP8MatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(x), axis=-1, dst_type=qdtype.x)
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-1, dst_type=qdtype.w)
        output = torch_npu.npu_quant_matmul(x_mxfp8, weight_mxfp8.t(), weight_scale.transpose(0, 1),
                                            pertoken_scale=x_scale,
                                            output_dtype=x.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        if len(x.shape) != 2:
            output = output.reshape(*x.shape[:-1], *output.shape[1:])
        if weight.requires_grad:
            output.requires_grad = True
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grads: torch.Tensor):
        x, weight = ctx.saved_tensors
        qdtype = get_quant_dtype()
        grads_mxfp8, grads_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(grads), axis=-1, dst_type=qdtype.grads)
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-2, dst_type=qdtype.w)
        dx = torch_npu.npu_quant_matmul(grads_mxfp8, weight_mxfp8, weight_scale,
                                        pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        grads_mxfp8, grads_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(grads), axis=-2, dst_type=qdtype.grads)
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(x), axis=-2, dst_type=qdtype.x)
        dw = torch_npu.npu_quant_matmul(grads_mxfp8.t(), x_mxfp8, x_scale, pertoken_scale=grads_scale.transpose(0, 1),
                                        output_dtype=x.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        return dx, dw, None, None
