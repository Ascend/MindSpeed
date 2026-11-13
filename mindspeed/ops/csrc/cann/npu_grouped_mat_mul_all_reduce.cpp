// Copyright 2024 Huawei Technologies Co., Ltd.
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

static const int64_t IN_NOT_SPLIT_OUT_NOT_SPLIT = 0;
static const int64_t IN_SPLIT_OUT_NOT_SPLIT = 1;
static const int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
static const int64_t IN_SPLIT_OUT_SPLIT = 3;

using npu_preparation = at_npu::native::OpPreparation;
#ifdef __TORCH_2__
    using BiasType = c10::optional<std::vector<at::Tensor>>;
#else
    using BiasType = std::vector<at::Tensor>;
#endif

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

bool _check_w_dim(size_t num_w, size_t dim_num_w, size_t dim_0_w, size_t num_group_list,
                  size_t sum_group_list)
{
    bool result = false;
    if (2 == dim_num_w && num_w == num_group_list) {
        result = true;
    } else if (3 == dim_num_w && 1 == num_w && dim_0_w == num_group_list) {
        result = true;
    } else if (2 == dim_num_w && 1 == num_w && dim_0_w == sum_group_list) {
        result = true;
    }
    return result;
}

void _check_dims(int64_t split_item, size_t num_x, size_t num_w, const at::TensorList &x,
                 const at::TensorList &weight, size_t num_group_list, size_t sum_group_list)
{
    TORCH_CHECK(num_x > 0 && num_w > 0,
        "Neither x nor weight could be empty.");
    TORCH_CHECK(IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item
        || IN_SPLIT_OUT_NOT_SPLIT == split_item || IN_SPLIT_OUT_SPLIT == split_item,
        "The given split_item [", split_item, "] is invalid, which must be one of 0/1/2/3");
    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item || IN_NOT_SPLIT_OUT_SPLIT == split_item) {
        TORCH_CHECK(num_x == num_w && 0 == num_group_list,
            "When split_item = 0 or 2, the num of x tensors must equal the num of weight tensors, "
            "and there is supposed not to be group_list input");
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item) {
        TORCH_CHECK(num_x == 1 && num_w == num_group_list && sum_group_list == x[0].sizes()[0],
            "When split_item = 1, the num of x tensors must equal 1, "
            "and the num of weight tensors is supposed to equal the length of group_list");
    } else if (IN_SPLIT_OUT_SPLIT == split_item) {
        size_t dim_num_w = weight[0].sizes().size();
        size_t dim_0_w = weight[0].sizes()[0];
        TORCH_CHECK(_check_w_dim(num_w, dim_num_w, dim_0_w, num_group_list, sum_group_list),
            "Invalid dim of weight. When split_item = 3, only the following three situations are allowed:"
            "(1) The tensor nums of weight equals the length of group_list; the dim num of each tensor equals 2. "
            "(2) There is one tensor in weight with a dim num of 3; its first dim equals the length of group_list. "
            "(3) There is one tensor in weight with a dim num of 2; its first dim equals the sum of group_list. ");
    }
}

void _create_new_tensor_multi_dim(std::vector<at::Tensor> &y, const at::Tensor &x_i,
                                  const at::Tensor &w_i, c10::TensorOptions options)
{
    auto x_sizes = x_i.sizes();
    std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
    y_sizes.at(x_sizes.size() - 1) = w_i.sizes()[1];
    y.emplace_back(at::empty(y_sizes, options));
}

void _create_new_tensor(std::vector<at::Tensor> &y, size_t dim_m, size_t dim_n, c10::TensorOptions options,
                        size_t num_group_list)
{
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(at::empty(output_size, options));
}

std::vector<at::Tensor> npu_grouped_mat_mul_all_reduce(const std::vector<at::Tensor>& x,
                                                       const std::vector<at::Tensor>& weight,
                                                       const BiasType& bias,
                                                       c10::optional<std::vector<int64_t>> group_list,
                                                       c10::optional<int64_t> split_item,
                                                       std::string hccl_group,
                                                       std::string reduce_op,
                                                       int64_t comm_turn)
{
    size_t num_x = x.size();
    size_t num_w = weight.size();
#ifdef __TORCH_2__
    const std::vector<at::Tensor>& new_bias = bias.value_or(std::vector<at::Tensor>{});
#else
    const std::vector<at::Tensor>& new_bias = bias;
#endif
    size_t num_bias = new_bias.size();

    const at::TensorList x_(x);
    const at::TensorList weight_(weight);
    const at::TensorList new_bias_(new_bias);

    auto group_list_real_ = group_list.value_or(std::vector<int64_t>{});
    at::IntArrayRef group_list_real(group_list_real_);
    size_t num_group_list = group_list_real.size();
    int64_t sum_group_list = num_group_list > 0 ? group_list_real[num_group_list - 1] : 0;
    int64_t split_item_value = split_item.value_or(0);

    const char* hccl_group_value = hccl_group.c_str();
    const char* reduce_op_value = reduce_op.c_str();

    int64_t comm_turn_value = comm_turn;
    int64_t stream_mode_value = 1;

    _check_dims(split_item_value, num_x, num_w, x_, weight_, num_group_list, sum_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_[0].options().dtype(x_[0].scalar_type());

    if (IN_NOT_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_x);
        for (size_t i = 0; i < num_x; i++) {
            _create_new_tensor_multi_dim(y, x[i], weight[i], options);
        }
    } else if (IN_SPLIT_OUT_NOT_SPLIT == split_item_value) {
        y.reserve(num_group_list);
        _create_new_tensor(y, group_list_real[0], weight[0].sizes()[1], options, num_group_list);
        for (size_t i = 1; i < num_group_list; i++) {
            _create_new_tensor(y, group_list_real[i] - group_list_real[i - 1], weight[i].sizes()[1], options,
                               num_group_list);
        }
    } else if (IN_NOT_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_m = 0;
        for (size_t i = 0; i < num_x; i++) {
            dim_m += x[i].sizes()[0];
        }
        _create_new_tensor(y, dim_m, weight[0].sizes()[1], options, num_group_list);
    } else if (IN_SPLIT_OUT_SPLIT == split_item_value) {
        size_t dim_num_w = weight[0].sizes().size();
        _create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1], options, num_group_list);
    }

    at::TensorList result(y);

    ACLNN_CMD(aclnnGroupedMatMulAllReduce, x_, weight_, new_bias_,
              group_list_real, split_item_value, hccl_group_value,
              reduce_op_value, comm_turn_value, stream_mode_value, result);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_grouped_mat_mul_all_reduce", &npu_grouped_mat_mul_all_reduce, "grouped mat_mul all_reduce forward");
}
