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


std::vector<at::Tensor> npu_ring_attention_update(const at::Tensor& prev_attn_out,
                                                  const at::Tensor& prev_softmax_max,
                                                  const at::Tensor& prev_softmax_sum,
                                                  const at::Tensor& cur_attn_out,
                                                  const at::Tensor& cur_softmax_max,
                                                  const at::Tensor& cur_softmax_sum,
                                                  c10::optional<at::Tensor>& actual_seq_qlen,
                                                  c10::optional<std::string> layout)
{
    auto softmax_max_shape = prev_softmax_max.sizes();
    auto softmax_sum_shape = prev_softmax_sum.sizes();
    auto attn_out_shape = prev_attn_out.sizes();

    auto softmax_max = at::empty(softmax_max_shape, prev_softmax_max.options());
    auto softmax_sum = at::empty(softmax_sum_shape, prev_softmax_sum.options());
    auto attn_out = at::empty(attn_out_shape, prev_attn_out.options());
    std::string layout_value = layout.value_or("SBH");
    char* input_layout_ptr = const_cast<char*>(layout_value.c_str());
    at::Tensor actual_seq_qlen_value = actual_seq_qlen.has_value() ? actual_seq_qlen.value() : at::empty(
        {}, prev_softmax_sum.options().dtype(at::ScalarType::Long));
    ACLNN_CMD(aclnnRingAttentionUpdate, prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
        cur_softmax_max, cur_softmax_sum, actual_seq_qlen_value, input_layout_ptr, attn_out, softmax_max, softmax_sum);
    return {attn_out, softmax_max, softmax_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_ring_attention_update", &npu_ring_attention_update, "npu ring attention update",
          pybind11::arg("prev_attn_out"),
          pybind11::arg("prev_softmax_max"),
          pybind11::arg("prev_softmax_sum"),
          pybind11::arg("cur_attn_out"),
          pybind11::arg("cur_softmax_max"),
          pybind11::arg("cur_softmax_sum"),
          pybind11::arg("actual_seq_qlen") = pybind11::none(),
          pybind11::arg("layout") = "SBH");
}