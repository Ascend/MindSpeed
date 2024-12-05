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

#include "../flop_counter/flop_counter.h"
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

void _check_dims(size_t num_x, const at::TensorList &weight, size_t num_group_list)
{
    size_t num_w = weight.size();
    TORCH_CHECK(num_x > 0 && num_w > 0,
        "Neither x nor weight could be empty.");
    size_t dim_num_w = weight[0].sizes().size();
    size_t dim_0_w = weight[0].sizes()[0];
}

void _create_new_tensor(std::vector<at::Tensor> &y, int64_t dim_m, int64_t dim_n, c10::TensorOptions options,
                        int64_t group_type_value, int64_t num_group_list)
{
    auto output_size = (2 == group_type_value) ? op_infer::array_to_small_vector({num_group_list, dim_m, dim_n})
                                               : op_infer::array_to_small_vector({dim_m, dim_n});
    y.emplace_back(at::empty(output_size, options));
}

void _foreach_transpose(const at::TensorList &tensorList, std::vector<at::Tensor> &tensors)
{
    for (int i = 0; i< tensorList.size(); i++) {
        at::Tensor tensor = tensorList[i].transpose(-1, -2);
        tensors.emplace_back(tensor);
    }
}

bool _is_transposed(at::Tensor &tensors)
{
    int dim_sum = tensors.dim();
    TORCH_CHECK(dim_sum >= 2 && dim_sum <= 3, // 2/3: gmm weight only support 2- or 3-dimensional
        "input tensor of is_tensor_transposed should be either 2- or 3-dimensional.");
    int shape_dim = tensors.sizes().size() - 2;
    if (tensors.stride(dim_sum - 2) == 1 && tensors.stride(dim_sum - 1) == tensors.sizes().at(shape_dim)) {
        return true;
    } else {
        return false;
    }
}

std::vector<at::Tensor> npu_gmm(const std::vector<at::Tensor>& x,
                                const std::vector<at::Tensor>& weight,
                                const std::vector<at::Tensor>& bias,
                                c10::optional<std::vector<int64_t>> group_list,
                                c10::optional<int64_t> group_type,
                                c10::optional<int64_t> group_list_type)
{
    auto num_x = x.size();
    auto num_w = weight.size();
    auto group_list_real_ = group_list.value_or(std::vector<int64_t>{});
    at::IntArrayRef group_list_real(group_list_real_);
    auto num_group_list = group_list_real.size();
    int64_t split_item_value = 3;
    int64_t group_type_value = group_type.value_or(-1);

    const at::TensorList x_(x);
    const at::TensorList weight_(weight);
    const at::TensorList bias_(bias);

    _check_dims(num_x, weight_, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_[0].options().dtype(x_[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    _create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1], options, group_type_value,
                        num_group_list);

    at::TensorList result = at::TensorList(y);
    auto scale_real = at::TensorList();
    auto offset_real = at::TensorList();
    auto antiquant_scale_real = at::TensorList();
    auto antiquant_offset_real = at::TensorList();
    ACLNN_CMD(aclnnGroupedMatmulV2, x_, weight_, bias_, scale_real, offset_real, antiquant_scale_real,
              antiquant_offset_real, group_list_real, split_item_value, group_type_value, result);
    #ifdef FLOP_COUNT
    FLOP_COUNT(FlopCounter::gmm_flop_int, x_, weight_, group_list, group_type_value);
    #endif
    return y;
}

std::vector<at::Tensor> npu_gmm(const std::vector<at::Tensor>& x,
                                const std::vector<at::Tensor>& weight,
                                const std::vector<at::Tensor>& bias,
                                const c10::optional<at::Tensor>& group_list,
                                c10::optional<int64_t> group_type,
                                c10::optional<int64_t> group_list_type)
{
    auto num_x = x.size();
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(at::Tensor());
    auto num_group_list = group_list_real.sizes()[0];
    int64_t split_item_value = 3;
    int64_t group_type_value = group_type.value_or(-1);
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t act_type_value = 0;

    const at::TensorList x_(x);
    const at::TensorList weight_(weight);
    const at::TensorList bias_(bias);

    _check_dims(num_x, weight_, num_group_list);

    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_[0].options().dtype(x_[0].scalar_type());

    size_t dim_num_w = weight[0].sizes().size();
    _create_new_tensor(y, x[0].sizes()[0], weight[0].sizes()[dim_num_w - 1], options, group_type_value,
                        num_group_list);

    at::TensorList result = at::TensorList(y);
    auto scale_real = at::TensorList();
    auto offset_real = at::TensorList();
    auto antiquant_scale_real = at::TensorList();
    auto antiquant_offset_real = at::TensorList();
    auto perToken_scale_real = at::TensorList();
    auto activation_input_real = at::TensorList();
    auto activation_quant_scale_real = at::TensorList();
    auto activation_quant_offset_real = at::TensorList();
    auto activation_feature_out_real = at::TensorList();
    auto dynQuant_scale_out_real = at::TensorList();

    ACLNN_CMD(aclnnGroupedMatmulV4, x_, weight_, bias_, scale_real, offset_real, antiquant_scale_real,
              antiquant_offset_real, perToken_scale_real, group_list_real, activation_input_real,
              activation_quant_scale_real, activation_quant_offset_real, split_item_value, group_type_value,
              group_list_type_value, act_type_value, result, activation_feature_out_real, dynQuant_scale_out_real);
    #ifdef FLOP_COUNT
    FLOP_COUNT(FlopCounter::gmm_flop_tensor, x_, weight_, group_list, group_type_value);
    #endif
    return y;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> npu_gmm_backward(
    const std::vector<at::Tensor>& grad,
    const std::vector<at::Tensor>& x,
    const std::vector<at::Tensor>& weight,
    const c10::optional<std::vector<int64_t>> group_list,
    c10::optional<int64_t> group_list_type)
{
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(std::vector<int64_t>{});

    const at::TensorList x_(x);
    const at::TensorList weight_(weight);

    std::vector<at::Tensor> xt;
    std::vector<at::Tensor> wt;

    _foreach_transpose(x_, xt);
    _foreach_transpose(weight_, wt);

    std::vector<at::Tensor> bias_real;

    std::vector<at::Tensor> dx = npu_gmm(grad, wt, bias_real, group_list_real, 0, group_list_type);
    std::vector<at::Tensor> dw = npu_gmm(xt, grad, bias_real, group_list_real, 2, group_list_type);
    std::vector<at::Tensor> dbias;

    std::vector<at::Tensor> dw_output;
    for (int i = 0; i < num_w; i++) {
        at::Tensor dw_tensor = dw[i].reshape(weight[i].sizes());
        dw_output.emplace_back(dw_tensor);
    }

    return std::make_tuple(dx, dw_output, dbias);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> npu_gmm_backward(
    const std::vector<at::Tensor>& grad,
    const std::vector<at::Tensor>& x,
    const std::vector<at::Tensor>& weight,
    const c10::optional<at::Tensor>& group_list,
    c10::optional<int64_t> group_list_type)
{
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(at::Tensor());

    std::vector<at::Tensor> bias_real;
    at::TensorList weight_(weight);
    std::vector<at::Tensor> wt;
    _foreach_transpose(weight_, wt);
    std::vector<at::Tensor> dx = npu_gmm(grad, wt, bias_real, group_list_real, 0, group_list_type);

    at::Tensor weight_tensor = weight.at(0);
    bool is_weight_transposed = _is_transposed(weight_tensor);

    std::vector<at::Tensor> dw;
    if (is_weight_transposed == true) {
        at::Tensor grad_tensor = grad.at(0).contiguous();
        at::TensorList grad_(grad_tensor);
        std::vector<at::Tensor> gradt;
        _foreach_transpose(grad_, gradt);
        std::vector<at::Tensor> dwt = npu_gmm(gradt, x, bias_real, group_list_real, 2, group_list_type);
        at::TensorList dwt_(dwt);
        _foreach_transpose(dwt_, dw);
    } else {
        at::TensorList x_(x);
        std::vector<at::Tensor> xt;
        _foreach_transpose(x_, xt);
        dw = npu_gmm(xt, grad, bias_real, group_list_real, 2, group_list_type);
    }

    std::vector<at::Tensor> dbias;
    std::vector<at::Tensor> dw_output;
    for (int i = 0; i < num_w; i++) {
        at::Tensor dw_tensor = dw[i].reshape(weight[i].sizes());
        dw_output.emplace_back(dw_tensor);
    }

    return std::make_tuple(dx, dw_output, dbias);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> npu_gmm_backward_fusion(
    const std::vector<at::Tensor>& grad,
    const std::vector<at::Tensor>& weight,
    const c10::optional<std::vector<int64_t>> group_list,
    c10::optional<int64_t> group_list_type)
{
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(std::vector<int64_t>{});

    const at::TensorList weight_(weight);

    std::vector<at::Tensor> wt;

    _foreach_transpose(weight_, wt);

    std::vector<at::Tensor> bias_real;

    std::vector<at::Tensor> dx = npu_gmm(grad, wt, bias_real, group_list_real, 0, group_list_type);

    std::vector<at::Tensor> dbias;

    std::vector<at::Tensor> dw_output;

    return std::make_tuple(dx, dw_output, dbias);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> npu_gmm_backward_fusion(
    const std::vector<at::Tensor>& grad,
    const std::vector<at::Tensor>& weight,
    const c10::optional<at::Tensor>& group_list,
    c10::optional<int64_t> group_list_type)
{
    auto num_w = weight.size();
    auto group_list_real = group_list.value_or(at::Tensor());

    std::vector<at::Tensor> bias_real;
    at::TensorList weight_(weight);
    std::vector<at::Tensor> wt;
    _foreach_transpose(weight_, wt);
    std::vector<at::Tensor> dx = npu_gmm(grad, wt, bias_real, group_list_real, 0, group_list_type);

    std::vector<at::Tensor> dbias;
    std::vector<at::Tensor> dw_output;

    return std::make_tuple(dx, dw_output, dbias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using gmmv1 = std::vector<at::Tensor>(*)(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, c10::optional<std::vector<int64_t>>, c10::optional<int64_t>, c10::optional<int64_t>);
    using gmmv2 = std::vector<at::Tensor>(*)(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const c10::optional<at::Tensor>&, c10::optional<int64_t>, c10::optional<int64_t>);
    using gmmv1_backward = std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>(*)(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const c10::optional<std::vector<int64_t>>, c10::optional<int64_t>);
    using gmmv2_backward = std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>(*)(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const c10::optional<at::Tensor>&, c10::optional<int64_t>);
    using gmmv1_backward_fusion = std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>(*)(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const c10::optional<std::vector<int64_t>>, c10::optional<int64_t>);
    using gmmv2_backward_fusion = std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>(*)(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const c10::optional<at::Tensor>&, c10::optional<int64_t>);
    
    m.def("npu_gmm", (gmmv1)&npu_gmm, "grouped matmul forward with group_list type List[int]");
    m.def("npu_gmm_backward", (gmmv1_backward)&npu_gmm_backward, "grouped matmul backward with group_list type List[int]");
    m.def("npu_gmm_backward_fusion", (gmmv1_backward_fusion)&npu_gmm_backward_fusion, "grouped matmul backward with group_list type List[int]");
    m.def("npu_gmm", (gmmv2)&npu_gmm, "grouped matmul forward with group_list type Tensor");
    m.def("npu_gmm_backward", (gmmv2_backward)&npu_gmm_backward, "grouped matmul backward with group_list type Tensor");
    m.def("npu_gmm_backward_fusion", (gmmv2_backward_fusion)&npu_gmm_backward_fusion, "grouped matmul backward with group_list type Tensor");
}
