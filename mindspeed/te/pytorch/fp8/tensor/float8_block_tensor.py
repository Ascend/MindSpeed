# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch_npu

from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8Tensor2D
from mindspeed.te.pytorch.fp8 import MatmulKey


class Float8BlockTensor(Float8Tensor2D):
    def quant_matmul(self, other: 'Float8BlockTensor', is_rowwise, key: MatmulKey):
        x1, x1_scale = self.get_quant_data(is_rowwise[0])
        x2, x2_scale = other.get_quant_data(is_rowwise[1])
        output = torch_npu.npu_quant_matmul(
            x1, x2, x2_scale, pertoken_scale=x1_scale, output_dtype=self.dtype, group_sizes=[1, 128, 128]
        )
        self.release(x1, x1_scale)
        other.release(x2, x2_scale)
        return self.restore_reshape(other, output)
