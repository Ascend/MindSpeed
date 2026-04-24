#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/RandomOpAdapter.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"


at::Tensor npu_sparse_attn_shared_kv_metadata(
    const c10::optional<at::Tensor> &cuSeqLensQ,
    const c10::optional<at::Tensor> &sequsedOriKv,
    const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t batchSize,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenKv,
    int64_t oriTopk,
    int64_t cmpTopk,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const c10::optional<std::string> layoutQ,
    const c10::optional<std::string> layoutKv,
    bool hasOriKv,
    bool hasCmpKv
){
    char *layoutQPtr = const_cast<char*>(layoutQ.value_or("SBH").c_str());
    char *layoutKvPtr = const_cast<char*>(layoutKv.value_or("SBH").c_str());
    at::Tensor metadata = at::empty(1024, at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kInt));
    ACLNN_CMD(aclnnSparseAttnSharedkvMetadata, cuSeqLensQ, sequsedOriKv, sequsedCmpKv, sequsedQ, sequsedKv, numHeadsQ, numHeadsKv, headDim, batchSize,
        maxSeqLenQ, maxSeqLenKv, oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr,
        layoutKvPtr, hasOriKv, hasCmpKv, metadata);
    return metadata;
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_shared_kv(
    const at::Tensor &query,
    const c10::optional<at::Tensor> &oriKv,
    const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices,
    const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable,
    const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqLensQ,
    const c10::optional<at::Tensor> &cuSeqLensOriKv,
    const c10::optional<at::Tensor> &cuSeqLensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv,
    const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata,
    double softmaxScale,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const c10::optional<std::string> layoutQ,
    const c10::optional<std::string> layoutKv,
    bool returnSoftmaxLse
)
{
    std::string layoutq = layoutQ.value_or("SBH");
    std::string layoutkv = layoutKv.value_or("SBH");
    char *layoutQPtr = const_cast<char*>(layoutq.c_str());
    char *layoutKvPtr = const_cast<char*>(layoutkv.c_str());

    at::Tensor attnOutput = at::empty(query.sizes(), query.options());
    at::Tensor softmaxLseOut;
    if(returnSoftmaxLse){
        std::vector<int64_t> lse_sizes(query.sizes().begin(), query.sizes().end());
        lse_sizes.back() = 1;
        softmaxLseOut = at::empty(lse_sizes, query.options().dtype(c10::ScalarType::Float));
    }else{
        softmaxLseOut = at::Tensor();
    }
    ACLNN_CMD(aclnnSparseAttnSharedkv, query, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable,
        cmpBlockTable, cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv, sequsedQ, sequsedKv,sinks, metadata, softmaxScale,
        cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr, returnSoftmaxLse, attnOutput,
        softmaxLseOut);
    return std::make_tuple(attnOutput, softmaxLseOut);
}

std::tuple<at::Tensor, at::Tensor, const c10::optional<at::Tensor>, at::Tensor> npu_sparse_attn_shared_kv_grad(
    const at::Tensor &query,
    const at::Tensor &oriKv,
    const c10::optional<const at::Tensor> &cmpKvOptional,
    const c10::optional<const at::Tensor> &dOutOptional,
    const c10::optional<const at::Tensor> &outOptional,
    const c10::optional<const at::Tensor> &lseOptional,
    const c10::optional<const at::Tensor> &oriSparseIndicesOptional,
    const c10::optional<const at::Tensor> &cmpSparseIndicesOptional,
    const c10::optional<const at::Tensor> &cuSeqlensQOptional,
    const c10::optional<const at::Tensor> &cuSeqlensOriKvOptional,
    const c10::optional<const at::Tensor> &cuSeqlensCmpKvOptional,
    const at::Tensor &sinks,
    double scaleValue,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    const c10::optional<std::string> layout
){
    const at::Tensor &cmpKv = cmpKvOptional.value_or(at::Tensor());
    const at::Tensor &dOut = dOutOptional.value_or(at::Tensor());
    const at::Tensor &out = outOptional.value_or(at::Tensor());
    const at::Tensor &lse = lseOptional.value_or(at::Tensor());
    const at::Tensor &oriSparseIndices = oriSparseIndicesOptional.value_or(at::Tensor());
    const at::Tensor &cmpSparseIndices = cmpSparseIndicesOptional.value_or(at::Tensor());
    const at::Tensor &cuSeqlensQ = cuSeqlensQOptional.value_or(at::Tensor());
    const at::Tensor &cuSeqlensOriKv = cuSeqlensOriKvOptional.value_or(at::Tensor());
    const at::Tensor &cuSeqlensCmpKv = cuSeqlensCmpKvOptional.value_or(at::Tensor());

    std::string layoutValue = layout.value_or("SBH");
    char *layoutPtr = const_cast<char*>(layoutValue.c_str());
    at::Tensor dQuery = at::empty(query.sizes(), query.options());
    at::Tensor dOriKv = at::empty(oriKv.sizes(), oriKv.options());
    at::Tensor dSinks = at::empty(sinks.sizes(), sinks.options());

    at::Tensor dCmpKv;
    if(cmpRatio > 1){
        dCmpKv = at::empty(cmpKv.sizes(), cmpKv.options());
    }else{
        dCmpKv = at::Tensor();
    }

    ACLNN_CMD(aclnnSparseAttnSharedkvGrad, query, oriKv, cmpKv, dOut, out, lse, oriSparseIndices, cmpSparseIndices,
        cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sinks, scaleValue, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
        oriWinRight, layoutPtr, dQuery, dOriKv, dCmpKv, dSinks);
    return std::make_tuple(dQuery, dOriKv, dCmpKv, dSinks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_sparse_attn_shared_kv_metadata", &npu_sparse_attn_shared_kv_metadata, "npu_sparse_attn_shared_kv metadata");
    m.def("npu_sparse_attn_shared_kv", &npu_sparse_attn_shared_kv, "npu_sparse_attn_shared_kv forward");
    m.def("npu_sparse_attn_shared_kv_grad", &npu_sparse_attn_shared_kv_grad, "npu_sparse_attn_shared_kv_grad backward");
}