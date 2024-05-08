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
#include "atb/infer_op_params.h"
#endif


void matmul_all_reduce(const at::Tensor &input1, const at::Tensor &input2, const c10::optional<at::Tensor> &biasOpt,
                       at::Tensor &output, int rank)
{
    const at::Tensor &bias = biasOpt.value_or(at::Tensor());

    atb::infer::LinearParallelParam param;
    bool transB = input1.size(1) != input2.size(0);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = 8;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;
    param.keepIntermediate = false;

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2);
    if (biasOpt.has_value()) {
        paramsetter.Input(bias);
    }
    paramsetter.Output(output);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "lcal coc get op failed!");
    RunAtbCmd(op, paramsetter, "matmul_all_reduce");
}


void all_gather_matmul(const at::Tensor &input1, const at::Tensor &input2, const c10::optional<at::Tensor> &biasOpt,
                       at::Tensor &output, int rank)
{
    const at::Tensor &bias = biasOpt.value_or(at::Tensor());

    atb::infer::LinearParallelParam param;
    bool transB = input1.size(1) != input2.size(0);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = 8;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR;
    param.keepIntermediate = false;

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2);
    if (biasOpt.has_value()) {
        paramsetter.Input(bias);
    }
    paramsetter.Output(output);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "lcal coc get op failed!");
    RunAtbCmd(op, paramsetter, "all_gather_matmul");
}


void all_gather_matmul_v2(const at::Tensor &input1, const at::Tensor &input2, const c10::optional<at::Tensor> &biasOpt,
                          at::Tensor &output, at::Tensor &commOutput, int rank)
{
    const at::Tensor &bias = biasOpt.value_or(at::Tensor());

    atb::infer::LinearParallelParam param;
    bool transB = input1.size(1) != input2.size(0);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = 8;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR;
    param.keepIntermediate = true;

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2);
    if (biasOpt.has_value()) {
        paramsetter.Input(bias);
    }
    paramsetter.Output(output)
               .Output(commOutput);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "lcal coc get op failed!");
    RunAtbCmd(op, paramsetter, "all_gather_matmul_v2");
}


void matmul_reduce_scatter(const at::Tensor &input1, const at::Tensor &input2, const c10::optional<at::Tensor> &biasOpt,
                           at::Tensor &output, int rank)
{
    const at::Tensor &bias = biasOpt.value_or(at::Tensor());

    atb::infer::LinearParallelParam param;
    bool transB = input1.size(1) != input2.size(0);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = 8;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER;
    param.keepIntermediate = false;

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2);
    if (biasOpt.has_value()) {
        paramsetter.Input(bias);
    }
    paramsetter.Output(output);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "lcal coc get op failed!");
    RunAtbCmd(op, paramsetter, "matmul_reduce_scatter");
}


void pure_matmul(const at::Tensor &input1, const at::Tensor &input2, const c10::optional<at::Tensor> &biasOpt,
                 at::Tensor &output, int rank)
{
    const at::Tensor &bias = biasOpt.value_or(at::Tensor());

    atb::infer::LinearParallelParam param;
    bool transB = input1.size(1) != input2.size(0);
    param.transWeight = transB;
    param.rank = rank;
    param.rankSize = 8;
    param.rankRoot = 0;
    param.hasResidual = biasOpt.has_value();
    param.backend = "lcoc";
    param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
    param.type = atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR;
    param.keepIntermediate = false;

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2);
    if (biasOpt.has_value()) {
        paramsetter.Input(bias);
    }
    paramsetter.Output(output);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "lcal coc get op failed!");
    RunAtbCmd(op, paramsetter, "pure_matmul");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_all_reduce", &matmul_all_reduce, "matmul_all_reduce", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"));
    m.def("all_gather_matmul", &all_gather_matmul, "all_gather_matmul", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"));
    m.def("all_gather_matmul_v2", &all_gather_matmul_v2, "all_gather_matmul_v2", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("commOutput"),
          pybind11::arg("rank"));
    m.def("matmul_reduce_scatter", &matmul_reduce_scatter, "matmul_reduce_scatter", pybind11::arg("input1"),
          pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"));
    m.def("pure_matmul", &pure_matmul, "pure_matmul", pybind11::arg("input1"), pybind11::arg("input2"),
          pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"));
}
