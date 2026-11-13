// Copyright 2024 Huawei Technologies Co., Ltd.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

const static int MIN_DIM = 2;
const static int X_MAX_DIM = 8;

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

at::Tensor npu_ffn(const at::Tensor &x, const at::Tensor &weight1, const at::Tensor &weight2,
    std::string activation, c10::optional<at::Tensor> expert_tokens, c10::optional<at::Tensor> expert_tokens_index,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &bias2,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &deq_scale1, const c10::optional<at::Tensor> &deq_scale2,
    const c10::optional<at::Tensor> &antiquant_scale1, const c10::optional<at::Tensor> &antiquant_scale2,
    const c10::optional<at::Tensor> &antiquant_offset1, const c10::optional<at::Tensor> &antiquant_offset2,
    c10::optional<int64_t> inner_precise, c10::optional<at::ScalarType> output_dtype)
{
    auto weight1_dim_num = weight1.dim();
    auto weight2_dim_num = weight2.dim();
    auto x_dim_num = x.dim();
    TORCH_CHECK(x_dim_num >= MIN_DIM && x_dim_num <= X_MAX_DIM, "x shape dims should be 2~8, but it is ", x_dim_num);
    auto x_k_dim = x.size(x.dim() - 1);
    auto wight1_k_dim = weight1.size(weight1.dim() - 2);
    TORCH_CHECK(x_k_dim == wight1_k_dim, "The k of x and weight should be equal. but x_k_dim is ",
        x_k_dim, ", wight1_k_dim is ", wight1_k_dim);

    TORCH_CHECK(!(expert_tokens.has_value() && expert_tokens_index.has_value()),
        "expert_tokens and expert_tokens_index should not have the value simultaneously.");

    char *activation_ptr = const_cast<char *>(activation.data());
    const at::Tensor &bias1_real = bias1.value_or(at::Tensor());
    const at::Tensor &bias2_real = bias2.value_or(at::Tensor());
    const at::Tensor &scale_real = scale.value_or(at::Tensor());
    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &deq_scale1_real = deq_scale1.value_or(at::Tensor());
    const at::Tensor &deq_scale2_real = deq_scale2.value_or(at::Tensor());
    const at::Tensor &antiquant_scale1_real = antiquant_scale1.value_or(at::Tensor());
    const at::Tensor &antiquant_scale2_real = antiquant_scale2.value_or(at::Tensor());
    const at::Tensor &antiquant_offset1_real = antiquant_offset1.value_or(at::Tensor());
    const at::Tensor &antiquant_offset2_real = antiquant_offset2.value_or(at::Tensor());
    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[x.dim() - 1] = weight2.size(weight2.dim() - 1);
    c10::TensorOptions options = x.options().dtype(x.scalar_type());
    if (x.scalar_type() == at::kChar && weight1.scalar_type() == at::kChar && weight2.scalar_type() == at::kChar) {
        options = x.options().dtype(output_dtype.value_or(at::kHalf));
    }
    at::Tensor result = at::empty(output_size, options);
    int64_t inner_precise_val = inner_precise.has_value() ? inner_precise.value() : 0;

    bool tokens_index_flag = expert_tokens_index.has_value();

    const at::Tensor &expert_tokens_real = expert_tokens.has_value() ? expert_tokens.value() :
        expert_tokens_index.value_or(at::Tensor());

    ACLNN_CMD(aclnnFFNV3, x, weight1, weight2, expert_tokens_real, bias1_real, bias2_real,
        scale_real, offset_real, deq_scale1_real, deq_scale2_real, antiquant_scale1_real, antiquant_scale2_real,
        antiquant_offset1_real, antiquant_offset2_real, activation_ptr, inner_precise_val, tokens_index_flag, result);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_ffn", &npu_ffn, "npu_ffn");
}