import torch_npu

from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8TensorWithTranspose
from mindspeed.te.pytorch.utils import view_as_n_dim


class GroupwiseTensor(Float8TensorWithTranspose):
    def t(self):
        raise ValueError('GroupwiseTensor not support transpose')

    def quant_matmul(self, other, transpose=(False, False)):
        x1, x1_scale = map(view_as_n_dim, self.get_by_trans(transpose[0]))
        x2, x2_scale = map(view_as_n_dim, other.get_by_trans(transpose[1]))
        output = torch_npu.npu_quant_matmul(x1, x2, x2_scale, pertoken_scale=x1_scale, output_dtype=self.dtype,
                                            group_sizes=[1, 128, 128])
        return self.restore_reshape(output, transpose[0])
