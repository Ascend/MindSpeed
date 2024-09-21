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

enum class X_SHARD_TYPE : int64_t {
    ALLGATHER_IN_H = 0,
    ALLGATHER_IN_C,
};

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
}  // namespace op_infer

const static int DIMS = 3;
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_alltoall_allgather_bmm(
    const at::Tensor &x,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    const std::string &group_ep,
    int64_t group_ep_worldsize,
    const std::string &group_tp,
    int64_t group_tp_worldsize,
    c10::optional<int64_t> shard_type,
    c10::optional<int64_t> act_type,
    c10::optional<bool> need_allgather_out,
    c10::optional<bool> need_activation_feature)
{
    TORCH_CHECK(x.dim() == DIMS,
        "The dims of input x should be 3 dimensional, but got ", x.dim(), "-dimensional.");
    TORCH_CHECK(weight.dim() == DIMS,
        "The dims of input weight should be 3 dimensional, but got ", weight.dim(), "-dimensional.");
    const at::Tensor &bias_const = bias.value_or(at::Tensor());
    const int64_t shard_type_value = shard_type.value_or(0);
    const X_SHARD_TYPE x_shard_type = static_cast<X_SHARD_TYPE>(shard_type_value);
    const int64_t act_type_value = act_type.value_or(0);
    const bool need_allgather_out_value = need_allgather_out.value_or(false);
    const bool need_activation_feature_value = need_activation_feature.value_or(false);
    char *group_ep_ptr = const_cast<char*>(group_ep.c_str());
    char *group_tp_ptr = const_cast<char*>(group_tp.c_str());

    c10::TensorOptions yoptions = x.options().dtype(x.scalar_type());
    auto batch = weight.size(0);
    auto m = x.size(1) * group_ep_worldsize;
    if (x_shard_type == X_SHARD_TYPE::ALLGATHER_IN_C) {
        m *= group_tp_worldsize;
    }
    auto k = weight.size(1);
    auto n = weight.size(2);

    auto y1_output_size = op_infer::array_to_small_vector({batch, m, n});
    at::Tensor y1out = at::empty(y1_output_size, yoptions);
    at::Tensor y2out{nullptr};
    if (need_allgather_out_value) {
        auto y2_output_size = op_infer::array_to_small_vector({batch, m, k});
        y2out = at::empty(y2_output_size, yoptions);
    }
    at::Tensor y3out{nullptr};
    if (need_activation_feature_value) {
        y3out = at::empty(y1_output_size, yoptions);
    }

    ACLNN_CMD(aclnnAlltoAllAllGatherBatchMatMul, x, weight, bias_const,
              group_ep_ptr, group_tp_ptr, group_ep_worldsize, group_tp_worldsize,
              shard_type_value, act_type_value, y1out, y2out, y3out);
    return std::tie(y1out, y2out, y3out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_alltoall_allgather_bmm", &npu_alltoall_allgather_bmm, "npu_alltoall_allgather_bmm realize");
}