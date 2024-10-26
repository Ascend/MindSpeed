// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"
#include "inc/mc2_utils.h"

static constexpr uint64_t DIMS = 3;
static constexpr uint64_t E_DIM_IDX = 0;
static constexpr uint64_t C_DIM_IDX = 1;
static constexpr uint64_t H_DIM_IDX = 2;
enum class Y_SHARD_TYPE : int64_t {
    SHARD_IN_H = 0,
    SHARD_IN_C,
};

static void check_params_dim(const at::Tensor &x, const at::Tensor &weight,
                             const c10::optional<at::Tensor> &bias)
{
    TORCH_CHECK(x.dim() == DIMS, "x needs to be 3D, but got: ", x.dim(), "D");
    TORCH_CHECK(weight.dim() == DIMS, "weight needs to be 3D, but got: ", weight.dim(), "D");
    TORCH_CHECK(x.size(2) == weight.size(1),
                "The K-axis in the two inputs of Matmul must be equal, but in reality, the K-axis of x is ",
                x.size(2), " and the K-axis of weight is ", weight.size(1));
    if (bias.has_value()) {
        const at::Tensor &bias_const = bias.value_or(at::Tensor());
        TORCH_CHECK(bias_const.dim() == DIMS or bias_const.dim() == 2,
            "bias needs to be 2D or 3D, but got: ", bias_const.dim(), "D");
    }
}

at::Tensor npu_bmm_reducescatter_alltoall(
    const at::Tensor &x,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    std::string group_ep,
    int64_t group_ep_worldsize,
    std::string group_tp,
    int64_t group_tp_worldsize,
    c10::optional<int64_t> shard_type)
{
    check_params_dim(x, weight, bias);
    const at::Tensor &bias_const = bias.value_or(at::Tensor());
    const int64_t shard_type_value = shard_type.value_or(0);
    const Y_SHARD_TYPE  y_shard_type = static_cast<Y_SHARD_TYPE>(shard_type_value);
    char *group_ep_ptr = const_cast<char*>(group_ep.c_str());
    char *group_tp_ptr = const_cast<char*>(group_tp.c_str());

    TORCH_CHECK(group_ep_worldsize > 0, "group_ep_worldsize should over 0, now is: ", group_ep_worldsize);
    TORCH_CHECK(group_tp_worldsize > 0, "group_tp_worldsize should over 0, now is: ", group_tp_worldsize);
    TORCH_CHECK((shard_type_value == 0) || (shard_type_value == 1), "shard_type should be 0 or 1, now is: ", shard_type_value);

    auto e = x.size(0) * group_ep_worldsize;

    int64_t c = x.size(C_DIM_IDX) / group_ep_worldsize;
    auto out_c_size = (y_shard_type == Y_SHARD_TYPE::SHARD_IN_C) ? (c / group_tp_worldsize) : c;

    int64_t h = weight.size(H_DIM_IDX);
    auto out_h_size = (y_shard_type == Y_SHARD_TYPE::SHARD_IN_H) ? (h / group_tp_worldsize) : h;
    auto y_shape = {e, out_c_size, out_h_size};

    at::Tensor y = at::empty(y_shape, x.options());
    ACLNN_CMD(aclnnBatchMatMulReduceScatterAlltoAll, x, weight, bias_const, group_ep_ptr, group_tp_ptr,
              group_ep_worldsize, group_tp_worldsize, shard_type_value, y);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_bmm_reducescatter_alltoall", &npu_bmm_reducescatter_alltoall, "npu_bmm_reducescatter_alltoall realize");
}