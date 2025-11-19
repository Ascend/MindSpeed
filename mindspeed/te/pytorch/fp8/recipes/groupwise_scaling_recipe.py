import dataclasses

import torch
import torch_npu

from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling, BlockDim
from mindspeed.te.pytorch.fp8.tensor import is_fp8_tensor
from mindspeed.te.pytorch.fp8.tensor.groupwise_tensor import GroupwiseTensor
from mindspeed.te.pytorch.utils import view_as_n_dim, get_quant_dtype


class GroupwiseScalingRecipe(Recipe):
    left_block_dim = BlockDim(row_block_size=1, col_block_size=128)
    right_block_dim = BlockDim(row_block_size=128, col_block_size=128)

    def quantization(self, tensor: torch.Tensor, key=None):
        if tensor is None:
            return tensor
        if is_fp8_tensor(tensor):
            return tensor
        tensor_2d = view_as_n_dim(tensor)
        if key == 'inputs':
            y, scale = torch_npu.npu_dynamic_block_quant(tensor_2d, dst_type=self.quant_dtype, **self.left_block_dim)
            y = y.view(tensor.shape)
            y_t, scale_t = torch_npu.npu_dynamic_block_quant(tensor_2d.t(), dst_type=self.quant_dtype,
                                                             **self.right_block_dim)
            y_t, scale_t = y_t.t(), scale_t.t()
        elif key == 'weight':
            y, scale = torch_npu.npu_dynamic_block_quant(tensor_2d.t(), dst_type=self.quant_dtype,
                                                         **self.right_block_dim)
            y, scale = y.t(), scale.t()
            y_t, scale_t = torch_npu.npu_dynamic_block_quant(tensor_2d, dst_type=self.quant_dtype,
                                                             **self.right_block_dim)
            y_t, scale_t = y_t.t(), scale_t.t()
        elif key == 'grads':
            y, scale = torch_npu.npu_dynamic_block_quant(tensor_2d, dst_type=self.quant_dtype, **self.left_block_dim)
            y = y.view(tensor.shape)
            y_t, scale_t = torch_npu.npu_dynamic_block_quant(tensor_2d.t(), dst_type=self.quant_dtype,
                                                             **self.left_block_dim)
        else:
            raise ValueError(f"key:{key} not supported yet.")
        return GroupwiseTensor(self.fp8_format_dtype, y, scale, y_t, scale_t, tensor.dtype)


@dataclasses.dataclass
class GroupwiseBlockScaling(RecipeScaling):
    recipe = GroupwiseScalingRecipe


class GroupwiseMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(x), dst_type=qdtype.x,
                                                             **GroupwiseScalingRecipe.left_block_dim)
        w_quant, w_scale = torch_npu.npu_dynamic_block_quant(weight, dst_type=qdtype.w,
                                                             **GroupwiseScalingRecipe.right_block_dim)
        output = torch_npu.npu_quant_matmul(x_mxfp8, w_quant.t(), w_scale.transpose(0, 1), pertoken_scale=x_scale,
                                            output_dtype=x.dtype, group_sizes=[1, 128, 128])
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
        grads_quant, grads_scale = torch_npu.npu_dynamic_block_quant(
            view_as_n_dim(grads), dst_type=qdtype.grads, **GroupwiseScalingRecipe.left_block_dim)
        w_quant, w_scale = torch_npu.npu_dynamic_block_quant(
            weight.t(), dst_type=qdtype.w, **GroupwiseScalingRecipe.right_block_dim)
        dx = torch_npu.npu_quant_matmul(grads_quant, w_quant.t(), w_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, group_sizes=[1, 128, 128])
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        grads_quant, grads_scale = torch_npu.npu_dynamic_block_quant(
            view_as_n_dim(grads).t(), dst_type=qdtype.grads, **GroupwiseScalingRecipe.left_block_dim)
        x_quant, x_scale = torch_npu.npu_dynamic_block_quant(
            view_as_n_dim(x).t(), dst_type=qdtype.x, **GroupwiseScalingRecipe.right_block_dim)
        dw = torch_npu.npu_quant_matmul(grads_quant, x_quant.t(), x_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, group_sizes=[1, 128, 128])
        return dx, dw, None, None
