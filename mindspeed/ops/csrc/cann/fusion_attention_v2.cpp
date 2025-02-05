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
#include <torch/extension.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"
#include "../flop_counter/flop_counter.h"

const static int FLASH_THRESHOLD = 512;
const static int N = 32;
const static int64_t SOFTMAXMAX_LAST_DIMSHAPE = 8;
using namespace at_npu::native;

constexpr static int SIZE_8 = 8;

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

enum class SparseMode {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};
DropOutStatus get_dropout_status(double keep_prob)
{
    if (keep_prob == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (keep_prob == 1.) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
        return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

at::Tensor dropout_gen_mask(const at::Tensor &query, const at::Tensor &key, double keep_prob, int64_t head_num, const std::string &input_layout,
    bool gen_mask_parallel, bool sync, int64_t &seed, int64_t &offset, int64_t &numels)
{
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = query.size(0) * head_num * query.size(1) * key.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = query.size(1) * head_num * query.size(0) * key.size(0); // [B,N,S,S]
    } else if (input_layout == "BNSD") {
        numels = query.size(0) * query.size(1) * query.size(2) * key.size(2); // [B,N,S,S]
    } else if (input_layout == "BSND") {
        numels = query.size(0) * query.size(2) * query.size(1) * key.size(1); // [B,N,S,S]
    }
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += 32;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = pair.first;
        offset = pair.second;
        drop_mask = at_npu::native::npu_dropout_gen_mask(query, at::IntArrayRef{ numels }, 1 - keep_prob,
                                                         seed, offset, gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }
    return drop_mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_backward_v2(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    const std::string &input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t inner_precise,
    const c10::optional<std::vector<int64_t>> &prefix,
    const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
    const c10::optional<std::vector<int64_t>> &actual_seq_kvlen,
    const c10::optional<std::vector<int64_t>> &q_start_idx,
    const c10::optional<std::vector<int64_t>> &kv_start_idx,
    int64_t sparse_mode,
    int64_t pse_type)
{
    double scale = scale_value;

    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
    const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
    const at::Tensor &softmax_const = softmax_in.value_or(at::Tensor());
    const at::Tensor &attention_const = attention_in.value_or(at::Tensor());
    auto prefixN_tmp = prefix.value_or(std::vector<int64_t>{});
    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(std::vector<int64_t>{});
    auto ac_seq_kvlen_tmp = actual_seq_kvlen.value_or(std::vector<int64_t>{});
    auto q_start_idx_val_tmp = q_start_idx.value_or(std::vector<int64_t>{});
    auto kv_start_idx_val_tmp = kv_start_idx.value_or(std::vector<int64_t>{});

    c10::optional<at::IntArrayRef> prefixN(prefixN_tmp);
    c10::optional<at::IntArrayRef> ac_seq_qlen(ac_seq_qlen_tmp);
    c10::optional<at::IntArrayRef> ac_seq_kvlen(ac_seq_kvlen_tmp);
    c10::optional<at::IntArrayRef> q_start_idx_val(q_start_idx_val_tmp);
    c10::optional<at::IntArrayRef> kv_start_idx_val(kv_start_idx_val_tmp);

    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_value = format_trans(value);
    at::Tensor format_dy = format_trans(dy);

    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_drop_mask = format_trans(drop_mask_const);
    at::Tensor format_padding_mask = format_trans(padding_mask_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);
    at::Tensor format_softmax_max = format_trans(softmax_max_const);
    at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
    at::Tensor format_softmax = format_trans(softmax_const);
    at::Tensor format_attention = format_trans(attention_const);
    at::Tensor dq = at::empty(format_query.sizes(), format_query.options());
    at::Tensor dk = at::empty(format_key.sizes(), format_key.options());
    at::Tensor dv = at::empty(format_value.sizes(), format_value.options());
    char* input_layout_ptr = const_cast<char *>(input_layout.c_str());
    at::Tensor dpse;
    if (format_pse.defined()) {
        dpse = at::empty(format_pse.sizes(), format_pse.options());
    } else {
        dpse = at::empty({0}, query.options());
    }

    if (!ac_seq_qlen_tmp.empty() && !ac_seq_kvlen_tmp.empty()) {
        ACLNN_CMD(
            aclnnFlashAttentionUnpaddingScoreGradV2, format_query, format_key, format_value, format_dy,
            format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
            format_softmax_sum, format_softmax, format_attention, prefixN, ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val,
            scale_value, keep_prob, pre_tokens, next_tokens, head_num, input_layout_ptr, inner_precise, sparse_mode, pse_type,
            dq, dk, dv, dpse);
    } else {
        ACLNN_CMD(
            aclnnFlashAttentionScoreGradV2, format_query, format_key, format_value, format_dy,
            format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
            format_softmax_sum, format_softmax, format_attention, prefixN, q_start_idx_val, kv_start_idx_val, scale_value, keep_prob,
            pre_tokens, next_tokens, head_num, input_layout_ptr, inner_precise, sparse_mode, pse_type, dq, dk, dv, dpse);
    }

    if (!format_pse.defined()) {
        at::Tensor dpse_required;
        dpse = dpse_required;
    }
    #ifdef FLOP_COUNT
    FLOP_COUNT(FlopCounter::flash_attention_backward_flop, query, key, value, dy, head_num, input_layout, actual_seq_qlen, actual_seq_kvlen);
    #endif
    return std::make_tuple(dq, dk, dv, dpse);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_grad_v2(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    const std::string &input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    const c10::optional<std::vector<int64_t>> &prefix,
    const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
    const c10::optional<std::vector<int64_t>> &actual_seq_kvlen,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync,
    int64_t pse_type,
    const c10::optional<std::vector<int64_t>> &q_start_idx,
    const c10::optional<std::vector<int64_t>> &kv_start_idx)
{
    TORCH_CHECK(query.dim() == 3 || query.dim() == 4, "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional");
    TORCH_CHECK(key.dim() == 3 || key.dim() == 4, "The shapes of the input key should be 3 or 4 dimensional, but got ",
        key.dim(), "-dimensional");
    TORCH_CHECK(value.dim() == 3 || value.dim() == 4, "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional");
    TORCH_CHECK(dy.dim() == 3 || dy.dim() == 4, "The shapes of the input dy should be 3 or 4 dimensional, but got ", dy.dim(), "-dimensional");
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
    TORCH_CHECK(pse_type >= 0 && pse_type <= 3, "The pse_type value must be in range of [0, 3], but got ", pse_type);
    std::string input_layout_str = std::string(input_layout);
    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode);
    } else {
        TORCH_CHECK(sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::PREFIX_COMPRESS),
                    "The sparse_mode value must be in range of [0,6], but got ",
                    sparse_mode);
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD" ||
        input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ", input_layout);

    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += 32;
    at::Tensor drop_mask;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        drop_mask = at_npu::native::npu_dropout_gen_mask(query, at::IntArrayRef{ numels }, 1 - keep_prob,
                                                         seed, offset, gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }
    auto result = npu_fusion_attention_backward_v2(query,
        key, value, dy, head_num, input_layout_str, pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tokens,
        next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, q_start_idx, kv_start_idx, sparse_mode, pse_type);
    if (!sync && get_dropout_status(keep_prob) != DropOutStatus::DROPOUT_NONE) {
        c10::Device device = drop_mask.device();
        c10::impl::VirtualGuardImpl impl(device.type());
        impl.recordDataPtrOnStream(drop_mask.storage().data_ptr(), c10_npu::getCurrentNPUStream());
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_fusion_attention_v2(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, const std::string &input_layout,
    const c10::optional<at::Tensor> &pse_opt, const c10::optional<at::Tensor> &padding_mask_opt,
    const c10::optional<at::Tensor> &atten_mask_opt,
    double scale, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise,
    const c10::optional<std::vector<int64_t>> &prefix_opt, const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
    const c10::optional<std::vector<int64_t>> &actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync,
    int64_t pse_type, const c10::optional<std::vector<int64_t>> &q_start_idx, const c10::optional<std::vector<int64_t>> &kv_start_idx)
{
    const at::Tensor &pse = pse_opt.value_or(at::Tensor());
    const at::Tensor &padding_mask = padding_mask_opt.value_or(at::Tensor());
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());
    auto prefixN_tmp = prefix_opt.value_or(std::vector<int64_t>{});
    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(std::vector<int64_t>{});
    auto ac_seq_kvlen_tmp = actual_seq_kvlen.value_or(std::vector<int64_t>{});
    auto q_start_idx_val_tmp = q_start_idx.value_or(std::vector<int64_t>{});
    auto kv_start_idx_val_tmp = kv_start_idx.value_or(std::vector<int64_t>{});

    c10::optional<at::IntArrayRef> prefixN(prefixN_tmp);
    c10::optional<at::IntArrayRef> ac_seq_qlen(ac_seq_qlen_tmp);
    c10::optional<at::IntArrayRef> ac_seq_kvlen(ac_seq_kvlen_tmp);
    c10::optional<at::IntArrayRef> q_start_idx_val(q_start_idx_val_tmp);
    c10::optional<at::IntArrayRef> kv_start_idx_val(kv_start_idx_val_tmp);

    TORCH_CHECK(head_num > 0, "head_num must > 0, but got ", head_num);
    TORCH_CHECK(query.dim() == 3 || query.dim() == 4, "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional");
    TORCH_CHECK(key.dim() == 3 || key.dim() == 4, "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional");
    TORCH_CHECK(value.dim() == 3 || value.dim() == 4, "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional");
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
    TORCH_CHECK(pse_type >= 0 && pse_type <= 3, "The pse_type value must be in range of [0, 3], but got ", pse_type);
    std::string input_layout_str = std::string(input_layout);
    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode);
    } else {
        TORCH_CHECK(sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::PREFIX_COMPRESS),
                    "The sparse_mode value must be in range of [0,6], but got ",
                    sparse_mode);
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD" ||
                input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ", input_layout);

    int64_t B = 0;
    int64_t S0 = 0; // S for query
    int64_t S1 = 0; // S for key & value
    int64_t N = 0;
    int64_t D = 0;
    int64_t H = 0;
    int64_t T = 0;
    int64_t D2 = 0; // D2 for value head-dim
    c10::SmallVector<int64_t> atten_score_shape;

    if (input_layout_str == "BSH") {
        B = query.size(0);
        S0 = query.size(1);
        S1 = key.size(1);
        H = query.size(2);
        D = H / head_num;
        D2 = (!D || !key.size(2)) ? 0 : value.size(2) / (key.size(2) / D);
        atten_score_shape = {B, S0, head_num * D2};
    } else if (input_layout_str == "SBH") {
        B = query.size(1);
        S0 = query.size(0);
        S1 = key.size(0);
        H = query.size(2);
        D = H / head_num;
        D2 = (!D || !key.size(2)) ? 0 : value.size(2) / (key.size(2) / D);
        atten_score_shape = {S0, B, head_num * D2};
    } else if (input_layout_str == "BNSD") {
        B = query.size(0);
        N = query.size(1);
        S0 = query.size(2);
        S1 = key.size(2);
        D = query.size(3);
        D2 = value.size(3);
        atten_score_shape = {B, N, S0, D2};
    } else if (input_layout_str == "BSND") {
        B = query.size(0);
        N = query.size(2);
        S0 = query.size(1);
        S1 = key.size(1);
        D = query.size(3);
        D2 = value.size(3);
        atten_score_shape = {B, S0, N, D2};
    } else if (input_layout_str == "TND") {
        T = query.size(0);
        N = query.size(1);
        D = query.size(2);
        D2 = value.size(2);
        atten_score_shape = {T, N, D2};
    }

    double scale_value = scale;

    at::Tensor format_query = format_trans(query);
    at::Tensor attention_score = at::empty(atten_score_shape, query.options());
    at::Tensor format_key = format_trans(key);
    at::Tensor format_value = format_trans(value);

    at::Tensor format_pse = format_trans(pse);
    at::Tensor format_padding_mask = format_trans(padding_mask);
    at::Tensor format_atten_mask = format_trans(atten_mask);

    int64_t seed;
    int64_t offset;
    int64_t numels;
    //check
    for(size_t i = 0; i < ac_seq_qlen_tmp.size(); i++){
        TORCH_CHECK(ac_seq_qlen_tmp[i] <= 1000000 && ac_seq_kvlen_tmp[i] <= 1000000, "The sequence length should not greater than 1M, but got q", ac_seq_qlen_tmp[i],"kv", ac_seq_kvlen_tmp[i]);
    }
    
    if (input_layout_str == "TND" && ac_seq_qlen_tmp.size() == ac_seq_kvlen_tmp.size()) {
        numels = N;
        int64_t accum = ac_seq_qlen_tmp[0] * ac_seq_kvlen_tmp[0];
        for (size_t i = 1; i < ac_seq_qlen_tmp.size(); i++) {
            accum += ((ac_seq_qlen_tmp[i] - ac_seq_qlen_tmp[i - 1]) * (ac_seq_kvlen_tmp[i] - ac_seq_kvlen_tmp[i - 1]));
        }
        numels *= accum;
    }

    at::Tensor format_drop_mask = dropout_gen_mask(format_query, format_key, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels);

    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::Tensor softmax_out;

    if (input_layout_str != "TND") {
        softmax_max = at::empty({B, head_num, S0, SOFTMAXMAX_LAST_DIMSHAPE}, query.options().dtype(at::kFloat)); // [B, N, S0, 8]
        softmax_sum = at::empty({B, head_num, S0, SOFTMAXMAX_LAST_DIMSHAPE}, query.options().dtype(at::kFloat)); // [B, N, S0, 8]
    } else {
        softmax_max = at::empty({T, N, SOFTMAXMAX_LAST_DIMSHAPE}, query.options().dtype(at::kFloat)); // [T, N, 8]
        softmax_sum = at::empty({T, N, SOFTMAXMAX_LAST_DIMSHAPE}, query.options().dtype(at::kFloat)); // [T, N, 8]
    }
    softmax_out = at::empty({0}, query.options());

    char* input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    if (!ac_seq_qlen_tmp.empty() && !ac_seq_kvlen_tmp.empty()) {
        ACLNN_CMD(
            aclnnFlashAttentionVarLenScoreV2, format_query, format_key, format_value,
            format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN,
            ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val, scale, keep_prob, pre_tokens, next_tokens, head_num,
            input_layout_ptr, inner_precise, sparse_mode, pse_type, softmax_max, softmax_sum,
            softmax_out, attention_score);
    } else {
        ACLNN_CMD(
            aclnnFlashAttentionScoreV2, format_query, format_key, format_value,
            format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN, q_start_idx_val, kv_start_idx_val,
            scale, keep_prob, pre_tokens, next_tokens, head_num, input_layout_ptr, inner_precise,
            sparse_mode, pse_type, softmax_max, softmax_sum, softmax_out, attention_score);
    }
    if (!sync && get_dropout_status(keep_prob) != DropOutStatus::DROPOUT_NONE) {
        c10::Device device = format_drop_mask.device();
        c10::impl::VirtualGuardImpl impl(device.type());
        impl.recordDataPtrOnStream(format_drop_mask.storage().data_ptr(), c10_npu::getCurrentNPUStream());
    }
    #ifdef FLOP_COUNT
    FLOP_COUNT(FlopCounter::flash_attention_forward_flop, query, key, value, head_num, input_layout, actual_seq_qlen, actual_seq_kvlen);
    #endif
    return std::make_tuple(attention_score, softmax_max, softmax_sum, softmax_out,
        seed, offset, numels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_fusion_attention_v2", &npu_fusion_attention_v2, "fusion attention forward v2");
    m.def("npu_fusion_attention_grad_v2", &npu_fusion_attention_grad_v2, "fusion attention backward v2");
}
