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

using namespace at_npu::native;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace {
    const static int DIMS = 4;
    const static int D_INDEX = 3;
    const static int EVEN_DIM_CHECK = 2;
    const static int BROADCAST_LIMIT = 1024;
    const static int64_t ROTATE_HALF = 0;
    const static int64_t ROTATE_INTERLEAVED = 1;

    void CheckRopeFroward(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, int64_t mode)
    {
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16 ||
                        x.scalar_type() == at::ScalarType::Float,
                    "Input tensor x dtype [", x.scalar_type(),
                    "] is invalid, should be float32, float16 or bfloat16");
        TORCH_CHECK(cos.scalar_type() == at::ScalarType::Half || cos.scalar_type() == at::ScalarType::BFloat16 ||
                        cos.scalar_type() == at::ScalarType::Float,
                    "Input tensor cos dtype [", cos.scalar_type(),
                    "] is invalid, should be float32, float16 or bfloat16");
        TORCH_CHECK(sin.scalar_type() == at::ScalarType::Half || sin.scalar_type() == at::ScalarType::BFloat16 ||
                        sin.scalar_type() == at::ScalarType::Float,
                    "Input tensor sin dtype [", sin.scalar_type(),
                    "] is invalid, should be float32, float16 or bfloat16");
        TORCH_CHECK(x.dim() == DIMS,
                    "The dims of input x should be 4 dimensional, bug got ", x.dim(), "-dimensional.");
        TORCH_CHECK(cos.dim() == DIMS,
                    "The dims of input cos should be 4 dimensional, bug got ", cos.dim(), "-dimensional.");
        TORCH_CHECK(sin.dim() == DIMS,
                    "The dims of input sin should be 4 dimensional, bug got ", sin.dim(), "-dimensional.");
        TORCH_CHECK(x.sizes()[D_INDEX] % EVEN_DIM_CHECK == 0,
                    "The head_dim length of input must be an even number, but got ", x.sizes()[D_INDEX], ".");
        TORCH_CHECK(cos.sizes() == sin.sizes(), "The shape of input Tensor cos and sin should be same.");
        TORCH_CHECK(mode == ROTATE_HALF || mode == ROTATE_INTERLEAVED,
                    "The mode of rotate shoule be 0(rotate_half) or 1(rotate_interleaved), but got ", mode, ".");
    }

    void CheckRopeBackward(const at::Tensor &y_grad, const at::Tensor &cos, const at::Tensor &sin, int64_t mode)
    {
        TORCH_CHECK(y_grad.scalar_type() == at::ScalarType::Half || y_grad.scalar_type() == at::ScalarType::BFloat16 ||
                        y_grad.scalar_type() == at::ScalarType::Float,
                    "Input tensor y_grad dtype [", y_grad.scalar_type(),
                    "] is invalid, should be float32, float16 or bfloat16");
        TORCH_CHECK(cos.scalar_type() == at::ScalarType::Half || cos.scalar_type() == at::ScalarType::BFloat16 ||
                        cos.scalar_type() == at::ScalarType::Float,
                    "Input tensor cos dtype [", cos.scalar_type(),
                    "] is invalid, should be float32, float16 or bfloat16");
        TORCH_CHECK(sin.scalar_type() == at::ScalarType::Half || sin.scalar_type() == at::ScalarType::BFloat16 ||
                        sin.scalar_type() == at::ScalarType::Float,
                    "Input tensor sin dtype [", sin.scalar_type(),
                    "] is invalid, should be float32, float16 or bfloat16");
        TORCH_CHECK(y_grad.dim() == DIMS,
                    "The dims of input y_grad should be 4 dimensional, bug got ", y_grad.dim(), "-dimensional.");
        TORCH_CHECK(cos.dim() == DIMS,
                    "The dims of input cos should be 4 dimensional, bug got ", cos.dim(), "-dimensional.");
        TORCH_CHECK(sin.dim() == DIMS,
                    "The dims of input sin should be 4 dimensional, bug got ", sin.dim(), "-dimensional.");
        TORCH_CHECK(y_grad.sizes()[D_INDEX] % EVEN_DIM_CHECK == 0,
                    "The head_dim length of input must be an even number, but got ", y_grad.sizes()[D_INDEX], ".");
        TORCH_CHECK(cos.sizes() == sin.sizes(), "The shape of input Tensor cos and sin should be same.");
        TORCH_CHECK(mode == ROTATE_HALF || mode == ROTATE_INTERLEAVED,
                    "The mode of rotate shoule be 0(rotate_half) or 1(rotate_interleaved), but got ", mode, ".");
        // when need to compute dcos and dsin, B * N < 1024
        if (cos.requires_grad() == true && sin.requires_grad() == true) {
            bool check_support = true;
            int64_t broadcast_dim_num = 1;
            for (int64_t i = 0; i < y_grad.dim(); i++) {
                if (y_grad.sizes()[i] != cos.sizes()[i]) {
                    broadcast_dim_num = broadcast_dim_num * y_grad.sizes()[i];
                }
                if (broadcast_dim_num > BROADCAST_LIMIT) {
                    check_support = false;
                    break;
                }
            }
            TORCH_CHECK(check_support == true,
                        "The broadcast shape: [", broadcast_dim_num, "] > 1024 is too large, do not support in backward function.");
        }
    }

    class NPURotaryPositionEmbedding : public torch::autograd::Function<NPURotaryPositionEmbedding> {
    public:
        static at::Tensor forward(AutogradContext *ctx, const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, c10::optional<int64_t> mode)
        {
            at::AutoDispatchBelowADInplaceOrView guard;
            int64_t mode_value = mode.value_or(ROTATE_HALF);
            CheckRopeFroward(x, cos, sin, mode_value);

            at::Tensor y = at::empty(x.sizes(), x.options());
            ACLNN_CMD(aclnnRotaryPositionEmbedding, x, cos, sin, mode_value, y);

            if (cos.requires_grad() == true && sin.requires_grad() == true) {
                ctx->save_for_backward({x, cos, sin});
            } else {
                ctx->save_for_backward({at::Tensor(), cos, sin});
            }
            ctx->saved_data["mode"] = mode_value;
            return y;
        }

        static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_output)
        {
            auto mode_value = ctx->saved_data["mode"].toInt();
            auto saved_vars = ctx->get_saved_variables();
            auto dy = grad_output[0];
            auto x = saved_vars[0];
            auto cos = saved_vars[1];
            auto sin = saved_vars[2];
            CheckRopeBackward(dy, cos, sin, mode_value);

            at::Tensor dx = at::empty(dy.sizes(), dy.options());
            at::Tensor dcos, dsin;
            if (cos.requires_grad() == true && sin.requires_grad() == true) {
                dcos = at::empty(cos.sizes(), cos.options());
                dsin = at::empty(sin.sizes(), sin.options());
            } else {
                dcos = at::empty({0}, cos.options());
                dsin = at::empty({0}, sin.options());
            }
            ACLNN_CMD(aclnnRotaryPositionEmbeddingGrad, dy, cos, sin, x, mode_value, dx, dcos, dsin);
            return {dx, dcos, dsin, at::Tensor()};
        }
    };
} // namespace

at::Tensor npu_rotary_position_embedding(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, int64_t mode)
{
    return NPURotaryPositionEmbedding::apply(x, cos, sin, mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_rotary_position_embedding", &npu_rotary_position_embedding,
          "rotary position embedding, mode 0: GPT-NeoX style, mode 1: GPT-J style",
          pybind11::arg("x"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("mode") = 0);
}
