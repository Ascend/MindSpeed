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
#include <torch/csrc/autograd/custom_function.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/include/ops.h>
#include "inc/aclnn_common.h"
#include <iostream>

using namespace at_npu::native;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace {
    const static int DIMS = 2;
    const static int MIN_DIMS = 1;

    void CheckMoeTokenPermuteForward(
        const at::Tensor &tokens,
        const at::Tensor &indices,
        bool padded_mode = false
    )
    {
        if (padded_mode) {
            throw std::runtime_error("current version only support padded_mode is false");
        }
        // current version tokens only support bfloat16
        TORCH_CHECK(tokens.scalar_type() == at::ScalarType::BFloat16,
                    "Input tensor tokens dtype [", tokens.scalar_type(),
                    "] is invalid, should be bfloat16");
        TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long || indices.scalar_type() == at::ScalarType::Int,
                    "Input tensor indices dtype [", indices.scalar_type(),
                    "] is invalid, should be int64 or int32");
        TORCH_CHECK(tokens.dim() == DIMS,
                    "The dims of input tokens should be 2 dimensional, but got ", tokens.dim(), "-dimensional.");
        TORCH_CHECK(indices.dim() == DIMS || indices.dim() == MIN_DIMS,
                    "The dims of input indices should be 2 or 1 dimensional, but got ", indices.dim(), "-dimensional.");
    }

    void CheckMoeTokenPermuteBackward(const at::Tensor &grad_permuted_tokens)
    {
        // current version grad_permuted_tokens only support bfloat16
        TORCH_CHECK(grad_permuted_tokens.scalar_type() == at::ScalarType::BFloat16,
                    "Input tensor permuted_tokens_grad dtype [", grad_permuted_tokens.scalar_type(),
                    "] is invalid, should be bfloat16");
        TORCH_CHECK(grad_permuted_tokens.dim() == DIMS,
                    "The dims of input grad_permuted_tokens should be 2 dimensional, but got ", grad_permuted_tokens.dim(), "-dimensional.");
    }

    class NPUMoeTokenPermute : public torch::autograd::Function<NPUMoeTokenPermute> {
    public:
        static std::vector<at::Tensor> forward(
            AutogradContext *ctx,
            const at::Tensor &tokens,
            const at::Tensor &indices,
            c10::optional<int64_t> num_out_tokens,
            c10::optional<bool> padded_mode
        )
        {
            at::AutoDispatchBelowADInplaceOrView guard;
            int64_t num_out_tokens_value = num_out_tokens.value_or(0);
            bool padded_mode_vale = padded_mode.value_or(false);
            CheckMoeTokenPermuteForward(tokens, indices, padded_mode_vale);

            int64_t topk = (indices.dim() == 1) ? 1 : indices.size(1);
            int64_t flatten_size = indices.numel();
            int64_t actual_num_out_tokens = (num_out_tokens_value > 0) ? std::min(num_out_tokens_value, flatten_size) : num_out_tokens_value + flatten_size;
            // The sorted_indices actually implemented by the aclnn operator are different from the sorted_indices
            // output by the permute function of the megatron source code.
            // The actual sorted_indices implemented by the aclnn operator are not sliced.
            // current version sorted_indices only support dtype(at::kInt)
            at::Tensor sorted_indices = at::empty({flatten_size}, indices.options().dtype(at::kInt));
            at::Tensor permuted_tokens = at::empty({actual_num_out_tokens, tokens.size(1)}, tokens.options());

            ACLNN_CMD(aclnnMoeTokenPermute, tokens, indices, actual_num_out_tokens, padded_mode_vale, permuted_tokens, sorted_indices);

            ctx->save_for_backward({sorted_indices});

            ctx->saved_data["num_tokens"] = tokens.size(0);
            ctx->saved_data["num_topK"] = topk;
            ctx->saved_data["padded_mode"] = padded_mode_vale;

            return {permuted_tokens, sorted_indices};
        }

        static std::vector<at::Tensor> backward(
            AutogradContext *ctx,
            std::vector<at::Tensor>& grad_output
        )
        {
            auto grad_permuted_tokens = grad_output[0];
            auto saved_tensors = ctx->get_saved_variables();
            auto sorted_indices = saved_tensors[0];

            int64_t num_tokens = ctx->saved_data["num_tokens"].toInt();
            int64_t num_topK = ctx->saved_data["num_topK"].toInt();
            bool padded_mode = ctx->saved_data["padded_mode"].toBool();
            CheckMoeTokenPermuteBackward(grad_permuted_tokens);

            at::Tensor grad_tokens = at::empty({num_tokens, grad_permuted_tokens.size(1)}, grad_permuted_tokens.options());

            ACLNN_CMD(aclnnMoeTokenPermuteGrad, grad_permuted_tokens, sorted_indices, num_topK, padded_mode, grad_tokens);

            return {grad_tokens, at::Tensor(), at::Tensor(), at::Tensor()};
        }
    };
} // namespace

std::vector<at::Tensor> npu_moe_token_permute(
    const at::Tensor &tokens,
    const at::Tensor &indices,
    c10::optional<int64_t> num_out_tokens,
    c10::optional<bool> padded_mode
)
{
    return NPUMoeTokenPermute::apply(tokens, indices, num_out_tokens, padded_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_token_permute", &npu_moe_token_permute,
          "npu moe token permute",
          pybind11::arg("tokens"),
          pybind11::arg("indices"),
          pybind11::arg("num_out_tokens") = 0,
          pybind11::arg("padded_mode") = false);
}