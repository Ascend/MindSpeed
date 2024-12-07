#include <torch/extension.h>
#include <torch/script.h>
#include <torch_npu/csrc/include/ops.h>
#include <torch/torch.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "inc/aclnn_common.h"

at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
        return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>npu_apply_fused_ema_adamw(
    at::Tensor grad,
    at::Tensor var,
    at::Tensor m,
    at::Tensor v,
    at::Tensor s,
    at::Tensor step,
    c10::optional<double> lr,
    c10::optional<double> ema_decay,
    c10::optional<double> beta1,
    c10::optional<double> beta2,
    c10::optional<double> eps,
    c10::optional<int64_t> mode,
    c10::optional<bool> bias_correction,
    c10::optional<double> weight_decay)
{
    at::Tensor grad_ = format_trans(grad);
    at::Tensor var_ = format_trans(var);
    at::Tensor m_ = format_trans(m);
    at::Tensor v_ = format_trans(v);
    at::Tensor s_ = format_trans(s);
    at::Tensor step_ = format_trans(step);
    double lr_ = double(lr.value());
    double ema_decay_ = double(ema_decay.value());
    double beta1_ = double(beta1.value());
    double beta2_ = double(beta2.value());
    double eps_ = double(eps.value());
    int64_t mode_ = int64_t(mode.value());
    bool bias_correction_ = bool(bias_correction.value());
    double weight_decay_ = double(weight_decay.value());
    ACLNN_CMD(aclnnApplyFusedEmaAdam,
              grad_,
              var_,
              m_,
              v_,
              s_,
              step_,
              lr_,
              ema_decay_,
              beta1_,
              beta2_,
              eps_,
              mode_,
              bias_correction_,
              weight_decay_);
    return std::tie(var_, m_, v_, s_);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_apply_fused_ema_adamw",
          &npu_apply_fused_ema_adamw,
          "npu_apply_fused_ema_adamw",
          pybind11::arg("grad"),
          pybind11::arg("var"),
          pybind11::arg("m"),
          pybind11::arg("v"),
          pybind11::arg("s"),
          pybind11::arg("step"),
          pybind11::arg("lr") = 1e-3f,
          pybind11::arg("ema_decay") = 0.9999,
          pybind11::arg("beta1") = 0.9,
          pybind11::arg("beta2") = 0.999,
          pybind11::arg("eps") = 1e-8f,
          pybind11::arg("mode") = 1,
          pybind11::arg("bias_correction") = true,
          pybind11::arg("weight_decay") = 0.0);
}