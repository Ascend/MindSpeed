#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/RandomOpAdapter.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"

using namespace at_npu::native;
const static int DIMENSION_3D = 3;
const static int DIMENSION_4D = 4;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_lightning_indexer_grad_kl_loss_symint(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &query_index,
    const at::Tensor &key_index,
    const at::Tensor &weights,
    const at::Tensor &sparse_indices,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<std::vector<int64_t>> actual_seq_qlen,
    const c10::optional<std::vector<int64_t>> actual_seq_klen,
    double scale_value,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens,
    c10::optional<int64_t> cmp_ratio)
{
    
    const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
    const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());

    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(std::vector<int64_t>{});
    auto actual_seq_klen_tmp = actual_seq_klen.value_or(std::vector<int64_t>{});
    c10::optional<at::IntArrayRef> actual_seq_qlen_const(ac_seq_qlen_tmp);
    c10::optional<at::IntArrayRef> actual_seq_klen_const(actual_seq_klen_tmp);
    c10::string_view layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(3);
    int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);
    bool deterministic_const = true;
    const int64_t cmp_ratio_const = cmp_ratio.value_or(1);
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional");
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input query_rope should be 3 or 4 dimensional, but got ",
            query_rope_const.dim(), "-dimensional");
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional");
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input key_rope should be 3 or 4 dimensional, but got ",
            key_rope_const.dim(), "-dimensional");
    }
    at::Tensor d_query_index = at::zeros(query_index.sizes(), query_index.options());
    at::Tensor d_key_index = at::zeros(key_index.sizes(), key_index.options());
    at::Tensor d_weights = at::zeros(weights.sizes(), weights.options());
    at::Tensor loss = at::zeros({1}, query.options().dtype(at::kFloat));

    ACLNN_CMD(
        aclnnSparseLightningIndexerGradKLLoss, query, key, query_index, key_index, weights,
        sparse_indices, softmax_max_const, softmax_sum_const, query_rope_const, key_rope_const, actual_seq_qlen_const,
        actual_seq_klen_const, scale_value, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, deterministic_const,
        cmp_ratio_const, d_query_index, d_key_index, d_weights, loss);

    return std::make_tuple(d_query_index, d_key_index, d_weights, loss);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_sparse_lightning_indexer_grad_kl_loss", &npu_sparse_lightning_indexer_grad_kl_loss_symint, "npu_sparse_lightning_indexer_grad_kl_loss");
}