import dataclasses

import torch
import torch_npu

from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling, BlockDim
from mindspeed.te.pytorch.fp8.tensor import Float8Tensor
from mindspeed.te.pytorch.utils import view_as_n_dim, get_quant_dtype


class BlockScalingRecipe(Recipe):
    block_dim = BlockDim(row_block_size=128, col_block_size=128)

    def quantization(self, tensor, key=None):
        if tensor is None:
            return tensor
        y, scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(tensor), dst_type=self.quant_dtype, **self.block_dim)
        if y.shape != tensor.shape:
            y = y.view(tensor.shape)
        return Float8Tensor(y, self.fp8_format_dtype, 1 / scale, tensor.dtype)


@dataclasses.dataclass
class BlockRecipeScaling(RecipeScaling):
    recipe = BlockScalingRecipe


class BlockwiseMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(x), dst_type=qdtype.x,
                                                             **BlockScalingRecipe.block_dim)
        w_quant, w_scale = torch_npu.npu_dynamic_block_quant(weight, dst_type=qdtype.w, **BlockScalingRecipe.block_dim)
        output = torch_npu.npu_quant_matmul(x_mxfp8, w_quant.t(), w_scale.transpose(0, 1), pertoken_scale=x_scale,
                                            output_dtype=x.dtype, group_sizes=[128, 128, 128])
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
        grads_quant, grads_scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(grads), dst_type=qdtype.grads,
                                                                     **BlockScalingRecipe.block_dim)
        w_quant, w_scale = torch_npu.npu_dynamic_block_quant(weight.t(), dst_type=qdtype.w,
                                                             **BlockScalingRecipe.block_dim)
        dx = torch_npu.npu_quant_matmul(grads_quant, w_quant.t(), w_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, group_sizes=[128, 128, 128])
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        grads_quant, grads_scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(grads).t(), dst_type=qdtype.grads,
                                                                     **BlockScalingRecipe.block_dim)
        x_quant, x_scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(x).t(), dst_type=qdtype.x,
                                                             **BlockScalingRecipe.block_dim)
        dw = torch_npu.npu_quant_matmul(grads_quant, x_quant.t(), x_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, group_sizes=[128, 128, 128])
        return dx, dw, None, None
