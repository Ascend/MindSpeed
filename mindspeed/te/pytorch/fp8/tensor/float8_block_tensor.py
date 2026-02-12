import torch_npu

from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8Tensor2D


class Float8BlockTensor(Float8Tensor2D):

    def quant_matmul(self, other: 'Float8BlockTensor', is_rowwise):
        x1, x1_scale = self.get_quant_data(is_rowwise[0])
        x2, x2_scale = other.get_quant_data(is_rowwise[1])
        output = torch_npu.npu_quant_matmul(x1, x2, x2_scale, pertoken_scale=x1_scale, output_dtype=self.dtype,
                                            group_sizes=[1, 128, 128])
        # 使用完 后续就不会继续调用, 直接清理掉显存
        for tensor in (x1, x1_scale, x2, x2_scale):
            tensor.untyped_storage().resize_(0)
        return self.restore_reshape(other, output)
