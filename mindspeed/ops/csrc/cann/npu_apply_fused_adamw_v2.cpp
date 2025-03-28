#include <torch/extension.h>
#include <torch/script.h>
#include <torch_npu/csrc/include/ops.h>
#include <torch/torch.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "inc/aclnn_common.h"

void npu_apply_fused_adamw_v2(
    at::Tensor param,
    at::Tensor grad,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::Tensor max_exp_avg_sq,
    at::Tensor state_step,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize)
{
//    bool is_same_size = (param.size() == grad.size() &&
//                       param.size() == exp_avg.size() &&
//                       param.size() == exp_avg_sq.size() &&
//                       (max_exp_avg_sq.size() == 0 ||
//                       param.size() == max_exp_avg_sq.size()));
//    if (!is_same_size) {
//        TORCH_CHECK(false, "the size of tensor list should be same.");
//    }

    float lr_cast = static_cast<float>(lr);
    float beta1_cast = static_cast<float>(beta1);
    float beta2_cast = static_cast<float>(beta2);
    float weight_decay_cast = static_cast<float>(weight_decay);
    float eps_cast = static_cast<float>(eps);

    auto step = state_step.sub(1);
    // max_exp_avg_sqs is optional when amsgrad is false
    if (!amsgrad) {
        c10::optional<at::Tensor> null_max_exp;
        ACLNN_CMD(aclnnApplyAdamWV2, param, exp_avg, exp_avg_sq, null_max_exp, grad,
            step, lr_cast, beta1_cast, beta2_cast, weight_decay_cast, eps_cast, amsgrad, maximize);
    } else {
        ACLNN_CMD(aclnnApplyAdamWV2, param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad,
            step, lr_cast, beta1_cast, beta2_cast, weight_decay_cast, eps_cast, amsgrad, maximize);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_apply_fused_adamw_v2", &npu_apply_fused_adamw_v2, "npu_apply_fused_adamw_v2");
}
