import dataclasses

import torch

import torch_npu
from mindspeed.te.pytorch.fp8.constants import TensorKey
from mindspeed.te.pytorch.fp8.recipes.recipe import Recipe, RecipeScaling, BlockDim
from mindspeed.te.pytorch.fp8.tensor import is_fp8_tensor
from mindspeed.te.pytorch.fp8.tensor.float8_block_tensor import Float8BlockTensor
from mindspeed.te.pytorch.utils import view_as_n_dim, get_quant_dtype


class Float8BlockRecipe(Recipe):
    left_dim = BlockDim(row_block_size=1, col_block_size=128)
    right_dim = BlockDim(row_block_size=128, col_block_size=128)
    rowwise, colwise = False, True

    quant_dim: dict[tuple[TensorKey, bool], BlockDim] = {
        (TensorKey.inputs, rowwise): right_dim,
        (TensorKey.inputs, colwise): left_dim,
        (TensorKey.weight, rowwise): right_dim,
        (TensorKey.weight, colwise): right_dim,
        (TensorKey.grads, rowwise): left_dim,
        (TensorKey.grads, colwise): left_dim,
    }

    def quantization(self, tensor: torch.Tensor, key, colwise, rowwise):
        if tensor is None:
            return tensor
        if is_fp8_tensor(tensor):
            return tensor
        tensor_2d = view_as_n_dim(tensor)
        col_data, row_data, col_scale, row_scale = None, None, None, None
        quant_tensor = Float8BlockTensor(self.fp8_format_dtype, tensor.shape, tensor.dtype)

        col_quant_dim = self.quant_dim[(key, self.colwise)]
        row_quant_dim = self.quant_dim[(key, self.rowwise)]

        if colwise:
            col_data, col_scale = torch_npu.npu_dynamic_block_quant(
                tensor_2d, dst_type=self.quant_dtype, **col_quant_dim
            )
        if rowwise:
            row_data, row_scale = torch_npu.npu_dynamic_block_quant(
                tensor_2d.T if key == TensorKey.grads else tensor_2d, dst_type=self.quant_dtype, **row_quant_dim
            )

        if key == TensorKey.weight:
            col_data, row_data, col_scale, row_scale = \
                row_data, col_data, row_scale, col_scale
        
        quant_tensor.set_col_data(col_data, col_scale)
        quant_tensor.set_row_data(row_data, row_scale, key == TensorKey.weight)
        return quant_tensor


@dataclasses.dataclass
class Float8BlockScaling(RecipeScaling):
    recipe = Float8BlockRecipe


class Float8BlockMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        qdtype = get_quant_dtype()
        x_mxfp8, x_scale = torch_npu.npu_dynamic_block_quant(view_as_n_dim(x), dst_type=qdtype.x,
                                                             **Float8BlockRecipe.left_dim)
        w_quant, w_scale = torch_npu.npu_dynamic_block_quant(weight, dst_type=qdtype.w, **Float8BlockRecipe.right_dim)
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
            view_as_n_dim(grads), dst_type=qdtype.grads, **Float8BlockRecipe.left_dim)
        w_quant, w_scale = torch_npu.npu_dynamic_block_quant(
            weight.t(), dst_type=qdtype.w, **Float8BlockRecipe.right_dim)
        dx = torch_npu.npu_quant_matmul(grads_quant, w_quant.t(), w_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, group_sizes=[1, 128, 128])
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        grads_quant, grads_scale = torch_npu.npu_dynamic_block_quant(
            view_as_n_dim(grads).t(), dst_type=qdtype.grads, **Float8BlockRecipe.left_dim)
        x_quant, x_scale = torch_npu.npu_dynamic_block_quant(
            view_as_n_dim(x).t(), dst_type=qdtype.x, **Float8BlockRecipe.right_dim)
        dw = torch_npu.npu_quant_matmul(grads_quant, x_quant.t(), x_scale.transpose(0, 1), pertoken_scale=grads_scale,
                                        output_dtype=x.dtype, group_sizes=[1, 128, 128])
        return dx, dw, None, None
