#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/RandomOpAdapter.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"

const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t SIZE = 4;

std::tuple<at::Tensor, at::Tensor> construct_lightning_indexer_output_tensor(
    const at::Tensor& query,
    const at::Tensor& key, 
    const c10::optional<at::Tensor> &actual_seq_lengths_query, 
    int64_t sparse_count,
    std::string query_layout_str, 
    std::string key_layout_str)
{
    at::SmallVector<int64_t, SIZE> output_size;

    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    at::Tensor sparse_indices_out = at::empty(output_size, at::kInt);
    at::Tensor sparse_values_out = at::empty(output_size, query.dtype());

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer(
    const at::Tensor &query, 
    const at::Tensor &key, 
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, 
    c10::string_view layout_query,
    c10::string_view layout_key, 
    int64_t sparse_count, 
    int64_t sparse_mode,
    int64_t pre_tokens, 
    int64_t next_tokens, 
    int64_t cmp_ratio, 
    bool return_value)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    at::SmallVector<int64_t, SIZE> output_size;
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    if (query_layout_str == "BSND") { 
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count}; 
    } else {
        int n_dim_index = 0; 
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2; 
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count}; 
    }
    
    at::Tensor sparse_values_out = at::empty(output_size, query.options());
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));

    ACLNN_CMD(aclnnLightningIndexer, query, key, weights, 
              actual_seq_lengths_query, actual_seq_lengths_key, block_table,
              query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode, pre_tokens, next_tokens, cmp_ratio,
              return_value, sparse_indices_out, sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_lightning_indexer_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &dy,
    const at::Tensor &sparse_indices,
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<std::string> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens,
    c10::optional<int64_t> cmp_ratio)
{
    at::Tensor d_query = at::zeros(query.sizes(), query.options());
    at::Tensor d_key = at::zeros(key.sizes(), key.options());
    at::Tensor d_weights = at::zeros(weights.sizes(), weights.options());
    
    std::string layout_str_view = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str_view.c_str());
    const int64_t sparse_mode_const = sparse_mode.value_or(0);
    const int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    const int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);
    const int64_t cmp_ratio_const = cmp_ratio.value_or(1);
    const int64_t head_num = 64;
    const bool deterministic = false;

    ACLNN_CMD(aclnnLightningIndexerGrad, query, key, dy, sparse_indices, weights,
              actual_seq_lengths_query, actual_seq_lengths_key,
              head_num, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, deterministic, cmp_ratio_const,
              d_query, d_key, d_weights);
    return std::make_tuple(d_query, d_key, d_weights);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_lightning_indexer", &npu_lightning_indexer, "npu_lightning_indexer forward");
    m.def("npu_lightning_indexer_grad", &npu_lightning_indexer_grad, "npu_lightning_indexer_grad backward");
}