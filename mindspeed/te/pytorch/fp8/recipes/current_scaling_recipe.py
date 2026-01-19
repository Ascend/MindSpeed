import dataclasses

import torch
import torch_npu

from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling
from mindspeed.te.pytorch.fp8.tensor import is_fp8_tensor, Float8Tensor
from mindspeed.te.pytorch.utils import view_as_n_dim, get_quant_dtype


class CurrentScalingRecipe(Recipe):

    def quantization(self, tensor, key, colwise, rowwise):
        if tensor is None:
            return tensor
        if is_fp8_tensor(tensor):  # if dtype is fp8
            return tensor
        quant_tensor, scale = torch_npu.npu_dynamic_quant(tensor, dst_type=self.quant_dtype, quant_mode='pertensor')
        return Float8Tensor(quant_tensor, self.quant_dtype, scale, dtype=tensor.dtype)


@dataclasses.dataclass
class Float8CurrentScaling(RecipeScaling):
    recipe = CurrentScalingRecipe


class TensorwiseMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        qdtype = get_quant_dtype()
        x_quant, x_scale = torch_npu.npu_dynamic_quant(x, dst_type=qdtype.x, quant_mode='pertensor')
        w_quant, w_scale = torch_npu.npu_dynamic_quant(weight, dst_type=qdtype.w, quant_mode='pertensor')

        output = torch_npu.npu_quant_matmul(x_quant, w_quant.t(), w_scale, pertoken_scale=x_scale,
                                            output_dtype=x.dtype, **qdtype.mm_kwargs)
        if weight.requires_grad:
            output.requires_grad = True
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grads: torch.Tensor):
        x, weight = ctx.saved_tensors
        qdtype = get_quant_dtype()
        w_quant, w_scale = torch_npu.npu_dynamic_quant(weight, dst_type=qdtype.w, quant_mode='pertensor')
        grads_quant, grads_scale = torch_npu.npu_dynamic_quant(grads, dst_type=qdtype.grads, quant_mode='pertensor')
        dx = torch_npu.npu_quant_matmul(grads_quant, w_quant, w_scale, pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, **qdtype.mm_kwargs)

        x_quant, x_scale = torch_npu.npu_dynamic_quant(x, dst_type=qdtype.x, quant_mode='pertensor')
        grads_quant, grads_scale = torch_npu.npu_dynamic_quant(view_as_n_dim(grads).t(), dst_type=qdtype.grads,
                                                               quant_mode='pertensor')
        dw = torch_npu.npu_quant_matmul(grads_quant, view_as_n_dim(x_quant), x_scale, pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, **qdtype.mm_kwargs)
        return dx, dw, None, None
