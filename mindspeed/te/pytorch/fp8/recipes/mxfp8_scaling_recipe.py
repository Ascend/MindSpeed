import dataclasses

import torch

import torch_npu
from mindspeed.te.pytorch.fp8 import MXFP8Tensor
from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling
from mindspeed.te.pytorch.utils import view_as_n_dim, get_quant_dtype


class MXFP8ScalingRecipe(Recipe):

    def quantization(self, tensor: torch.Tensor, key=None):
        if tensor is None:
            return tensor
        ori_dtype = tensor.dtype
        axis = -1
        # 当前MXFP8的quant matmul只支持MK*NK的形式，因此axis轴均为-1轴K轴
        # input、weight、grad三者均需要将转置和非转置全部处理，因此这里无传参控制是否采集转置数据
        if key == 'weight':
            y, mx_scale = torch_npu.npu_dynamic_mx_quant(tensor, axis=-2, dst_type=self.fp8_format_dtype)
        else:
            y, mx_scale = torch_npu.npu_dynamic_mx_quant(tensor, axis=axis, dst_type=self.fp8_format_dtype)

        if key == 'inputs':
            y_t, mx_scale_t = torch_npu.npu_dynamic_mx_quant(tensor, axis=0, dst_type=self.fp8_format_dtype)
        elif key == 'grads':
            tensor_t = view_as_n_dim(tensor).t()
            y_t, mx_scale_t = torch_npu.npu_dynamic_mx_quant(tensor_t, axis=axis,
                                                             dst_type=self.fp8_format_dtype)
        else:
            y_t, mx_scale_t = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(tensor), axis=axis,
                                                             dst_type=self.fp8_format_dtype)
            y_t, mx_scale_t = y_t.t(), mx_scale_t.transpose(0, 1)
        return MXFP8Tensor(self.fp8_format_dtype, y, mx_scale, y_t, mx_scale_t, ori_dtype)


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
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight.t(), axis=-1, dst_type=qdtype.w)
        dx = torch_npu.npu_quant_matmul(grads_mxfp8, weight_mxfp8.t(), weight_scale.transpose(0, 1),
                                        pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        grads_mxfp8, grads_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(grads).t(), axis=-1,
                                                                  dst_type=qdtype.grads)
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(x).t(), axis=-1, dst_type=qdtype.x)
        dw = torch_npu.npu_quant_matmul(grads_mxfp8, x_mxfp8.t(), x_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, scale_dtype=torch_npu.float8_e8m0fnu,
                                        pertoken_scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32])
        return dx, dw, None, None
