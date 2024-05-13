// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#ifdef ENABLE_ATB
#include <torch_npu/csrc/core/npu/SecondaryStreamGuard.h>
#include <torch_npu/csrc/include/ops.h>
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#endif

using namespace std;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
namespace {
#ifdef ENABLE_ATB
const static int QKV_DIM_NUM = 3;
const static int QKV_DIM_NUM_BNSD = 4;
const static int BNSD_DIM_B = 0;
const static int BNSD_DIM_S = 2;
const static int BNSD_DIM_D = 3;
const static int BSH_DIM_B = 0;
const static int BSH_DIM_S = 1;
const static int BSH_DIM_H = 2;
const static int SBH_DIM_B = 1;
const static int SBH_DIM_S = 0;
const static int SBH_DIM_H = 2;
const static int N = 32;

enum FlashAttentionFormat : int {
    BNSD = 0,
    BSH,
    SBH
}; 

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

DropOutStatus get_status(double keep_prob)
{
    if (keep_prob == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (keep_prob == 1.) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

void gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
                   const int64_t offset, const int64_t numels, at::Tensor &mask)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    c10::TensorOptions options = self.options();
    mask = at::empty(at::IntArrayRef(length + 32), options.dtype(at::kByte));
    at::SmallVector<int64_t, N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef(numels))
        .Input(keep_prob, self.scalar_type(), at_npu::native::CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(seed, at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong, at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
}

// select gen_mask method
void gen_mask_dispatch(const at::Tensor &self, const double &keep_prob, const int64_t &seed,
                       const int64_t offset, const int64_t numels, const bool parallel, const bool sync, at::Tensor &mask)
{
    if (parallel) {
        mask = at_npu::native::npu_dropout_gen_mask(self, at::IntArrayRef(numels), keep_prob, seed, offset, parallel, sync);
    } else {
        gen_mask_impl(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels, mask);
    }
}

void InferShapeFlashAttention(c10::SmallVector<int64_t, N> &size, int64_t io_layout, int64_t head_num, const at::Tensor &query)
    {
        if (io_layout == BNSD) {
            // BNSD
            size = {query.size(BNSD_DIM_B), head_num, query.size(BNSD_DIM_S)};
        } else if (io_layout == BSH) {
            // BSH
            size = {query.size(BSH_DIM_B), head_num, query.size(BSH_DIM_S)};
        } else if (io_layout == SBH) {
            // SBH
            size = {query.size(SBH_DIM_B), head_num, query.size(SBH_DIM_S)};
        }
    }

void CheckFlashAttention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                         int64_t head_num, int64_t io_layout)
{
    TORCH_CHECK(query.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "Input Q dtype ", query.scalar_type(),
                " invalid, should be float16 or bfloat16");
    TORCH_CHECK(key.scalar_type() == at::ScalarType::Half || key.scalar_type() == at::ScalarType::BFloat16,
                "Input K dtype ", key.scalar_type(),
                " invalid, should be float16 or bfloat16");
    TORCH_CHECK(value.scalar_type() == at::ScalarType::Half || value.scalar_type() == at::ScalarType::BFloat16,
                "Input V dtype ", value.scalar_type(),
                " invalid, should be float16 or bfloat16");
    int dim_s;
    int dim_b;
    int dim_h;
    int dim_num = QKV_DIM_NUM;
    if (io_layout == BNSD) {
        // BNSD
        dim_s = BNSD_DIM_S;
        dim_b = BNSD_DIM_B;
        dim_h = BNSD_DIM_D;
        dim_num = QKV_DIM_NUM_BNSD;
    } else if (io_layout == BSH) {
        // BSH
        dim_s = BSH_DIM_S;
        dim_b = BSH_DIM_B;
        dim_h = BSH_DIM_H;
    } else if (io_layout == SBH) {
        // SBH
        dim_s = SBH_DIM_S;
        dim_b = SBH_DIM_B;
        dim_h = SBH_DIM_H;
    }
    TORCH_CHECK(
        query.dim() == dim_num,
        "Input Q dim num ", query.dim(), " invalid, should be ", dim_num);
    TORCH_CHECK(
        key.dim() == dim_num,
        "Input K dim num ", key.dim(), " invalid, should be ", dim_num);
    TORCH_CHECK(
        value.dim() == dim_num,
        "Input V dim num ", value.dim(), " invalid, should be ", dim_num);
    auto batch_size = query.size(dim_b);
    auto head_dim_size = query.size(dim_h);

    TORCH_CHECK(key.size(dim_b) == batch_size &&
                key.size(dim_h) == head_dim_size,
                "Shape of input Q and input K should be same in batch_size_dim and head_dim");
    TORCH_CHECK(value.size(dim_b) == batch_size &&
                value.size(dim_h) == head_dim_size,
                "Shape of input Q and input V should be same in batch_size_dim and head_dim");
    TORCH_CHECK(value.size(dim_s) == key.size(dim_s),
                "Shape of input K and input V should be same in batch_size_dim and head_dim");
}

void CheckFlashAttentionBackward(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum,
                                 const at::Tensor &attention_out, const at::Tensor &query,
                                 const at::Tensor &key, const at::Tensor &value,
                                 int64_t head_num, int64_t io_layout)
{   CheckFlashAttention(query, key, value, head_num, io_layout);
    TORCH_CHECK(dy.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "Input dy dtype ", dy.scalar_type(), " invalid, should be float16 or bfloat16");
    TORCH_CHECK(softmax_log_max_sum.scalar_type() == at::ScalarType::Float,
                "Input softmax_log_max_sum dtype ", softmax_log_max_sum.scalar_type(),
                " invalid, should be float ");
    TORCH_CHECK(attention_out.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "Input attention_out dtype ", attention_out.scalar_type(),
                " invalid, should be float16 or bfloat16");
}

// compute flash_attention_forward
void flash_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                     const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                     const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num,
                     int64_t io_layout, float keep_prob, int64_t pre_tokens, int64_t next_tokens,
                     int64_t precise_mode, int64_t groups, at::Tensor &tensor_softmax, at::Tensor &tensor_attention_out)
{
    atb::train::FlashAttentionParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    //set input and output
    ParamSetter paramsetter;
    paramsetter.Input(query)
               .Input(key)
               .Input(value)
               .Input(atten_mask)
               .Input(alibi_mask)
               .Input(drop_mask)
               .Output(tensor_attention_out)
               .Output(tensor_softmax);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "flash_attention get op failed!");
    RunAtbCmd(op, paramsetter, "fa_forward");
}

// compute flash_attention_backward
void flash_attention_grad(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum, const at::Tensor &attention_out,
                          const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                          const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                          const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num, int64_t io_layout,
                          float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups,
                          at::Tensor &tensor_query_grad, at::Tensor &tensor_key_grad, at::Tensor &tensor_value_grad)
{
    atb::train::FlashAttentionBackwardParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionBackwardParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    ParamSetter paramsetter;
    paramsetter.Input(dy)
               .Input(softmax_log_max_sum)
               .Input(attention_out)
               .Input(query)
               .Input(key)
               .Input(value)
               .Input(atten_mask)
               .Input(alibi_mask)
               .Input(drop_mask)
               .Output(tensor_query_grad)
               .Output(tensor_key_grad)
               .Output(tensor_value_grad);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "flash_attention get op failed!");
    RunAtbCmd(op, paramsetter, "fa_backward");
}

// gen mask for flash_attention_forward , get seed offset numels for backward to gen mask
void gen_mask(const at::Tensor &self, double keep_prob,
              int64_t head_num, int64_t input_layout, bool parallel,
              bool sync, int64_t &seed, int64_t &offset, int64_t &numels, at::Tensor &drop_mask)
{
    if (input_layout == BSH) {
        numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
    } else if (input_layout == SBH) {
        numels = self.size(1) * head_num * self.size(0) * self.size(0); // [B,N,S,S]
    } else if (input_layout == BNSD) {
        numels = self.size(0) * self.size(1) * self.size(2) * self.size(2); // [B,N,S,S]
    }
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = pair.first;
        offset = pair.second;
        gen_mask_dispatch(self, keep_prob, seed,
            offset, numels, parallel, sync, drop_mask);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        int64_t length = (numels + 256 - 1) / 256 * 256 / 8 + 32;
        drop_mask = at::zeros(at::IntArrayRef(length), self.options().dtype(at::kByte));
    }
}

// gen mask for flash_attention_backward , use seed offset numels generate mask
void gen_mask(const at::Tensor &self, double keep_prob, bool parallel, bool sync,
              int64_t seed, int64_t offset, int64_t numels, at::Tensor &drop_mask)
{
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        gen_mask_dispatch(self, keep_prob, seed, offset, numels,
                          parallel, sync, drop_mask);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        int64_t length = (numels + 256 - 1) / 256 * 256 / 8 + 32;
        drop_mask = at::zeros(at::IntArrayRef(length), self.options().dtype(at::kByte));
    }
}

int64_t transLayout(std::string io_layout)
{
    int64_t layout;
    if (io_layout == "BSH") {
        layout = BSH;
    } else if (io_layout == "SBH") {
        layout = SBH;
    } else if (io_layout == "BNSD") {
        layout = BNSD;
    } else {
        TORCH_CHECK(false, "The input_layout should be BSH/SBH/BNSD(case-insensitive), but got ", io_layout);
    }
    return layout;
}
#endif

class NPUFlashAttentionFunction : public torch::autograd::Function<NPUFlashAttentionFunction> {
public:
    static at::Tensor forward(
        AutogradContext *ctx, const at::Tensor &query, const at::Tensor &key,
        const at::Tensor &value, const c10::optional<at::Tensor> &atten_mask_opt,
        const c10::optional<at::Tensor> &alibi_mask_opt, float scale_value = 1.0,
        float q_scale = 1.0, int64_t head_num = 1, std::string io_layout = "BNSD",
        float keep_prob = 1.0, int64_t pre_tokens = 2147483647,
        int64_t next_tokens = 1, int64_t precise_mode = 0, int64_t groups = -1,
        bool sync = false, bool parallel = true)
    {
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "flash_attention not implemented");
#else
        at::AutoNonVariableTypeMode g;
        int64_t layout = transLayout(io_layout);
        CheckFlashAttention(query, key, value, head_num, layout);

        const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());
        const at::Tensor &alibi_mask = alibi_mask_opt.value_or(at::Tensor());

        at::Tensor drop_mask;
        int64_t seed;
        int64_t offset;
        int64_t numels;
        gen_mask(query, keep_prob, head_num, layout,
                 parallel, sync, seed, offset, numels, drop_mask);

        c10::SmallVector<int64_t, N> tensor_softmax_shape;
        InferShapeFlashAttention(tensor_softmax_shape, layout, head_num, query);
        // apply tensor
        at::Tensor tensor_softmax = at::empty(at::IntArrayRef(tensor_softmax_shape),
                                              query.options().dtype(at::ScalarType::Float));
        at::Tensor tensor_attention_out = at::empty(query.sizes(), query.options());

        flash_attention(query, key, value, atten_mask, alibi_mask, drop_mask,
                        scale_value, head_num, layout, keep_prob, pre_tokens, next_tokens,
                        precise_mode, groups, tensor_softmax, tensor_attention_out);
        
        ctx->save_for_backward({query, key, value, atten_mask, alibi_mask,
                                tensor_attention_out, tensor_softmax});
        ctx->saved_data["scale_value"] = scale_value;
        ctx->saved_data["q_scale"] = q_scale;
        ctx->saved_data["keep_prob"] = keep_prob;
        ctx->saved_data["pre_tokens"] = pre_tokens;
        ctx->saved_data["next_tokens"] = next_tokens;
        ctx->saved_data["head_num"] = head_num;
        ctx->saved_data["layout"] = layout;
        ctx->saved_data["parallel"] = parallel;
        ctx->saved_data["sync"] = sync;
        ctx->saved_data["groups"] = groups;
        ctx->saved_data["precise_mode"] = precise_mode;
        ctx->saved_data["seed"] = seed;
        ctx->saved_data["offset"] = offset;
        ctx->saved_data["numels"] = numels;

        return tensor_attention_out;
#endif
    }

    static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_output)
    {
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "flash_attention_grad not implemented");
#else
        auto scale_value = ctx->saved_data["scale_value"].toDouble();
        auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
        auto pre_tokens = ctx->saved_data["pre_tokens"].toInt();
        auto next_tokens = ctx->saved_data["next_tokens"].toInt();
        auto head_num = ctx->saved_data["head_num"].toInt();
        auto layout = ctx->saved_data["layout"].toInt();
        auto precise_mode = ctx->saved_data["precise_mode"].toInt();
        auto seed = ctx->saved_data["seed"].toInt();
        auto offset = ctx->saved_data["offset"].toInt();
        auto numels = ctx->saved_data["numels"].toInt();
        auto groups = ctx->saved_data["groups"].toInt();
        auto parallel = ctx->saved_data["parallel"].toBool();
        auto sync = ctx->saved_data["sync"].toBool();

        auto saved = ctx->get_saved_variables();
        auto query = saved[0];
        auto key = saved[1];
        auto value = saved[2];
        auto atten_mask = saved[3];
        auto alibi_mask = saved[4];
        auto attention_out = saved[5];
        auto softmax_log_max_sum = saved[6];
        CheckFlashAttentionBackward(grad_output[0], softmax_log_max_sum,
                                    attention_out, query,
                                    key, value,
                                    head_num, layout);
        at::Tensor drop_mask;
        gen_mask(query, keep_prob, parallel, sync,
                 seed, offset, numels, drop_mask);

        at::Tensor tensor_query_grad = at::empty(query.sizes(), query.options());
        at::Tensor tensor_key_grad = at::empty(key.sizes(), key.options());
        at::Tensor tensor_value_grad = at::empty(value.sizes(), value.options());

        flash_attention_grad(grad_output[0], softmax_log_max_sum, attention_out, query,
                             key, value, atten_mask, alibi_mask, drop_mask,
                             scale_value, head_num, layout, keep_prob,
                             pre_tokens, next_tokens, precise_mode, groups,
                             tensor_query_grad, tensor_key_grad,
                             tensor_value_grad);

        return {tensor_query_grad, tensor_key_grad, tensor_value_grad,
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
#endif
    }
};
} // namespace

at::Tensor npu_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
    float scale_value, float q_scale, int64_t head_num,
    std::string io_layout, float keep_prob, int64_t pre_tokens, int64_t next_tokens,
    int64_t precise_mode, int64_t groups,
    bool sync, bool parallel)
{
    return NPUFlashAttentionFunction::apply(query, key, value, atten_mask, alibi_mask, scale_value, q_scale,
        head_num, io_layout, keep_prob, pre_tokens, next_tokens, precise_mode, groups,
        sync, parallel);
}
