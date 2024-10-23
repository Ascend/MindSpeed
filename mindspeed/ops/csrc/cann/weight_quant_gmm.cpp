// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/RandomOpAdapter.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"

using npu_preparation = at_npu::native::OpPreparation;

namespace op_infer {
constexpr int SIZE = 8;

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> small_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        small_shape.emplace_back(shape[i]);
    }
    return small_shape;
}
}

std::vector<at::Tensor> npu_weight_quant_gmm(const std::vector<at::Tensor>& x,
                                             const std::vector<at::Tensor>& weight,
                                             const std::vector<at::Tensor>& antiquant_scale,
                                             const std::vector<at::Tensor>& antiquant_offset,
                                             const std::vector<at::Tensor>& bias,
                                             const c10::optional<at::Tensor>& group_list,
                                             c10::optional<int64_t> group_list_type,
                                             c10::optional<int64_t> act_type)
{
    auto group_list_real = group_list.value_or(at::Tensor());
    int64_t split_item_value = 3;
    int64_t group_type_value = 0;
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = act_type.value_or(0);

    const at::TensorList x_(x);
    const at::TensorList weight_(weight);
    const at::TensorList bias_(bias);
    const at::TensorList antiquant_scale_(antiquant_scale);
    const at::TensorList antiquant_offset_(antiquant_offset);

    c10::TensorOptions options = x_[0].options().dtype(x_[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    auto output_size = op_infer::array_to_small_vector({x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1]});
    std::vector<at::Tensor> y{at::empty(output_size, options)};
    at::TensorList result = at::TensorList(y);
    auto scale = nullptr;
    auto offset = nullptr;
    auto per_token_scale = nullptr;
    auto act_in = nullptr;
    auto act_quant_scale = nullptr;
    auto act_quant_offset = nullptr;
    auto act_out = nullptr;
    auto dynamic_quant_scale_out = nullptr;
    ACLNN_CMD(aclnnGroupedMatmulV4, x_, weight_, bias_, scale, offset, antiquant_scale_,
              antiquant_offset_, per_token_scale, group_list_real, act_in, act_quant_scale, act_quant_offset,
              split_item_value, group_type_value, group_list_type_value, act_type_value,
              result, act_out, dynamic_quant_scale_out);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_weight_quant_gmm", &npu_weight_quant_gmm, "weight quantize grouped matmul forward");
}
