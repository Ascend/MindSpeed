/******************************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * All rights reserved.
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <torch/extension.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include <torch_npu/csrc/include/ops.h>

void check_npu_mm_all_reduce_add_rms_norm_params(const at::Tensor &x1, const at::Tensor &x2,
                                                 const at::Tensor &residual,
                                                 const at::Tensor &gamma,
                                                 const c10::optional<at::Tensor> &antiquant_scale,
                                                 const c10::optional<at::Tensor> &antiquant_offset,
                                                 const c10::optional<at::Tensor> &dequant_scale)
{
    // check shape: shape of x1:[b,s,k], shape of x2:[k,n], shape of residual:[b,s,n], shape of gamma:[n],
    TORCH_CHECK(x1.dim() == 2 || x1.dim() == 3, "x1 needs to be 2D or 3D, but got: ", x1.dim(), "D");
    TORCH_CHECK(x2.dim() == 2, "x2 needs to be 2D, but got: ", x2.dim(), "D");
    TORCH_CHECK(residual.dim() == 3, "residual needs to be 3D, but got: ", residual.dim(), "D");
    TORCH_CHECK(gamma.dim() == 1, "gamma needs to be 1D, but got: ", gamma.dim(), "D");
    TORCH_CHECK(x1.size(x1.dim() - 1) == x2.size(0), "K of x1 and x2 should be same, but they are x1_k: ",
                x1.size(x1.dim() - 1), ", x2_k: ", x2.size(0));
    size_t x1_bs = x1.size(0);
    if (x1.dim() == 3) {
        x1_bs *= x1.size(1);
    }
    TORCH_CHECK(x1_bs == (residual.size(0) * residual.size(1)), "(b*s) of x1 and residual should be same,",
                "but they are x1_(b*s): ", x1_bs, ", residual_(b*s): ", (residual.size(0) * residual.size(1)));
    TORCH_CHECK(x2.size(x2.dim() - 1) == residual.size(residual.dim() - 1), "n of x2 and residual should be same,",
                "but they are x2_n: ", x2.size(x2.dim() - 1), ", residual_n: ", residual.size(residual.dim() - 1));
    TORCH_CHECK(residual.size(residual.dim() - 1) == gamma.size(0), "n of residual and gamma should be same,",
                "but they are residual_n: ", residual.size(residual.dim() - 1), ", gamma_n: ", gamma.size(0));

    // check parameters.
    // aclnn apis for MC2 share one torch_npu api, therefore, each aclnn api only accepts parameters
    // that will be used. Any unused parameter will be seen as illegal. The job must be done here in
    // torch_npu api.
    // A8W8: antiquantScale and antiquantOffset should be None.
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x1.scalar_type() == at::kChar, "x1 must be an int8 tensor for quant.");
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for quant.");
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value()),
                    "when both dtype of x1 and dtype of x2 are equal to int8, "
                    "antiquantScale, antiquantOffset should both be null");
    }
    // A16W8: dequantScale should be None.
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for weight quant.");
        TORCH_CHECK((!dequant_scale.has_value()),
                    "when only dtype of x2 is equal to int8, dequantScale should be null");
    }
    // MC2 without quantization. antiquantScale and antiquantOffset and dequantScale should be None.
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value() && !dequant_scale.has_value()),
                    "when neither dtype of x1 or dtype of x2 is equal to int8, "
                    "antiquantScale, antiquantOffset and dequantScale should all be null");
    }
}