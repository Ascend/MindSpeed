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
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "PluggableAllocator.h"
#include "NPUSwapManager.h"

extern "C" {
void *gmlake_malloc(size_t size, int device, aclrtStream stream)
{
    void *ptr = PluggableAllocator::getInstance().malloc(device, size, stream);
    return ptr;
}

void gmlake_free(void *ptr, size_t size, int device, aclrtStream stream)
{
    PluggableAllocator::getInstance().free(ptr);
}

void gmlake_init(int device_count)
{
    PluggableAllocator::getInstance().init(device_count);
}

void gmlake_empty_cache(bool check_error)
{
    PluggableAllocator::getInstance().emptyCache(true);
}

void gmlake_memory_fraction(double fraction, int device)
{
    PluggableAllocator::getInstance().setMemoryFraction(fraction, device);
}

DeviceStats gmlake_get_device_stats(int device)
{
    return PluggableAllocator::getInstance().getDeviceStats(device);
}

void gmlake_reset_peak_stats(int device)
{
    return PluggableAllocator::getInstance().resetPeakStats(device);
}

void gmlake_record_stream(void *ptr, c10_npu::NPUStream stream)
{
    PluggableAllocator::getInstance().recordStream(ptr, stream);
}

void gmlake_erase_stream(void *ptr, c10_npu::NPUStream stream)
{
    PluggableAllocator::getInstance().eraseStream(ptr, stream);
}
}

py::list small_vector_to_list(const c10::SmallVector<std::size_t, c10_npu::N> &sizes)
{
    py::list result;
    for (const auto &value : sizes) {
        result.append(value);
    }
    return result;
}

py::list getProfilerOpInfoData()
{
    py::list opList;
    for (auto &opInfo : c10_npu::swap::NPUSwapManager::GetInstance().getSwapProfiler()->getProfilerOpInfoVec()) {
        py::dict opDict;
        opDict["opName"] = opInfo.getOpName();
        opDict["opId"] = opInfo.getOpId();
        opDict["stage"] = opInfo.getStage();
        opDict["step"] = opInfo.getStep();
        opDict["allocated_bytes"] = opInfo.getSwapMemory().allocated_bytes;
        opDict["reserved_bytes"] = opInfo.getSwapMemory().reserved_bytes;
        opDict["active_bytes"] = opInfo.getSwapMemory().active_bytes;
        py::list tensorList;
        for (auto &tensorInfo : opInfo.getProfilerTensorInfo()) {
            py::dict tensorDict;
            tensorDict["ptr"] = tensorInfo.getPtr();
            tensorDict["size"] = tensorInfo.getNbytes();
            tensorDict["shape"] = small_vector_to_list(tensorInfo.getShapeV2());
            tensorDict["dtype"] = c10::toString(tensorInfo.getDtype());
            tensorDict["tensorType"] = tensorInfo.getTensorType();
            tensorList.append(tensorDict);
        }
        opDict["tensor"] = tensorList;
        opList.append(opDict);
    }
    return opList;
}

py::list getProfilerSwapInfoData()
{
    py::list opList;
    for (auto &opInfo : c10_npu::swap::NPUSwapManager::GetInstance().getSwapProfiler()->getProfilerSwapInfoVec()) {
        py::dict opDict;
        opDict["opId"] = opInfo.getOpId();
        opDict["swapName"] = opInfo.getSwapName();
        opDict["size"] = opInfo.getSize();
        opDict["isOOM"] = opInfo.getIsOOM();
        opDict["srcPtr"] = opInfo.getSrcPtr();
        opDict["dstPtr"] = opInfo.getDstPtr();
        opList.append(opDict);
    }
    return opList;
}

void setPolicyInfoData(std::vector<c10_npu::swap::SwapPolicyInfo> &policyInfoVec)
{
    c10_npu::swap::NPUSwapManager::GetInstance().FunAfterProfiler(policyInfoVec);
}

void setFrequentOpNameData(std::vector<std::string> &frequentOpNames)
{
    c10_npu::swap::NPUSwapManager::GetInstance().initOpNameToOneHotAndIndexMap(frequentOpNames);
}

void updateStep()
{
    c10_npu::swap::NPUSwapManager::GetInstance().updateStep();
}

void updateProfiler()
{
    c10_npu::swap::NPUSwapManager::GetInstance().getSwapProfiler()->updateStep();
}

std::vector<c10_npu::swap::UniqueSwapPtr> recordTensorPtrWithTypes(const std::vector<torch::Tensor> &tensors,
    c10_npu::swap::SwapTensorType tensorType, int updateWeakPtrMap, bool isUpdateBlacklist)
{
    auto uniquePtrs = c10_npu::swap::NPUSwapManager::GetInstance().recordTensorPtrWithTypes(tensors, tensorType,
        updateWeakPtrMap, isUpdateBlacklist);
    return uniquePtrs;
}

void InitCppManager()
{
    c10_npu::swap::NPUSwapManager::GetInstance().Init();
}

void DeInitCppManager()
{
    c10_npu::swap::NPUSwapManager::GetInstance().DeInit();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<c10_npu::swap::SwapTensorType>(m, "SwapTensorType")
        .value("MODEL", c10_npu::swap::SwapTensorType::MODEL)
        .value("OPTIM", c10_npu::swap::SwapTensorType::OPTIM)
        .value("SHARED_MEMORY", c10_npu::swap::SwapTensorType::SHARED_MEMORY)
        .value("OTHERS", c10_npu::swap::SwapTensorType::OTHERS)
        .value("RESERVED", c10_npu::swap::SwapTensorType::RESERVED);

    py::enum_<c10_npu::swap::SwapStageType>(m, "SwapStageType")
        .value("INIT", c10_npu::swap::SwapStageType::INIT)
        .value("FWD", c10_npu::swap::SwapStageType::FWD)
        .value("BWD", c10_npu::swap::SwapStageType::BWD)
        .value("OPTIM", c10_npu::swap::SwapStageType::OPTIM)
        .value("RESERVED", c10_npu::swap::SwapStageType::RESERVED);

    py::class_<c10_npu::swap::UniqueSwapPtr>(m, "UniqueSwapPtr")
        .def(py::init<>())
        .def_readwrite("ptrBase", &c10_npu::swap::UniqueSwapPtr::ptrBase)
        .def_readwrite("index", &c10_npu::swap::UniqueSwapPtr::index);

    py::class_<c10_npu::swap::UniqueSwapMemory>(m, "UniqueSwapMemory")
        .def(py::init<>())
        .def_readwrite("allocated_bytes", &c10_npu::swap::UniqueSwapMemory::allocated_bytes)
        .def_readwrite("reserved_bytes", &c10_npu::swap::UniqueSwapMemory::reserved_bytes)
        .def_readwrite("active_bytes", &c10_npu::swap::UniqueSwapMemory::active_bytes);

    py::class_<c10_npu::swap::SwapStage>(m, "SwapStage")
        .def(py::init<>())
        .def_readwrite("stageType", &c10_npu::swap::SwapStage::stageType)
        .def_readwrite("microBatchIndex", &c10_npu::swap::SwapStage::microBatchIndex)
        .def_readwrite("layerIndex", &c10_npu::swap::SwapStage::layerIndex);

    py::class_<c10_npu::swap::SwapConfig>(m, "SwapConfig")
        .def(py::init<>()) // 测试用，可删除
        .def_readwrite("microBatchNum", &c10_npu::swap::SwapConfig::microBatchNum)
        .def_readwrite("layerNum", &c10_npu::swap::SwapConfig::layerNum)
        .def_readwrite("isOOM", &c10_npu::swap::SwapConfig::isOOM)
        .def_readwrite("stage", &c10_npu::swap::SwapConfig::stage)
        .def_readwrite("step", &c10_npu::swap::SwapConfig::step)
        .def_readwrite("policyStep", &c10_npu::swap::SwapConfig::policyStep)
        .def_readwrite("currentStageOpId", &c10_npu::swap::SwapConfig::currentStageOpId)
        .def_readwrite("oneStepDuration", &c10_npu::swap::SwapConfig::oneStepDuration)
        .def_readwrite("tensorSizeThresh", &c10_npu::swap::SwapConfig::tensorSizeThresh)
        .def_readwrite("fwdOpLayerInfo", &c10_npu::swap::SwapConfig::fwdOpLayerInfo)
        .def_readwrite("bwdOpLayerInfo", &c10_npu::swap::SwapConfig::bwdOpLayerInfo)
        .def_readwrite("enableProfiler", &c10_npu::swap::SwapConfig::enableProfiler)
        .def_readwrite("enableExecutor", &c10_npu::swap::SwapConfig::enableExecutor)
        .def_readwrite("enableCustomRecordStream", &c10_npu::swap::SwapConfig::enableCustomRecordStream);

    py::class_<c10_npu::swap::SwapPolicyInfo>(m, "SwapPolicyInfo")
        .def(py::init<>())
        .def_readwrite("ptr", &c10_npu::swap::SwapPolicyInfo::ptr)
        .def_readwrite("executorNeedMatch", &c10_npu::swap::SwapPolicyInfo::executorNeedMatch)
        .def_readwrite("swapOutOpId", &c10_npu::swap::SwapPolicyInfo::swapOutOpId)
        .def_readwrite("swapInOpId", &c10_npu::swap::SwapPolicyInfo::swapInOpId)
        .def_readwrite("swapOutStage", &c10_npu::swap::SwapPolicyInfo::swapOutStage)
        .def_readwrite("swapInStage", &c10_npu::swap::SwapPolicyInfo::swapInStage)
        .def_readwrite("freeStage", &c10_npu::swap::SwapPolicyInfo::freeStage)
        .def_readwrite("swapInFreeStage", &c10_npu::swap::SwapPolicyInfo::swapInFreeStage);

    py::class_<c10_npu::swap::NPUSwapManager>(m, "NPUSwapManager")
        .def_static("GetInstance", &c10_npu::swap::NPUSwapManager::GetInstance, py::return_value_policy::reference)
        .def_readwrite("config", &c10_npu::swap::NPUSwapManager::config)
        .def_readwrite("swap_enable", &c10_npu::swap::NPUSwapManager::swap_enable)
        .def_readwrite("swap_oom_enable", &c10_npu::swap::NPUSwapManager::swap_oom_enable);

    m.def("getProfilerOpInfoData", &getProfilerOpInfoData);
    m.def("getProfilerSwapInfoData", &getProfilerSwapInfoData);
    m.def("setPolicyInfoData", &setPolicyInfoData);
    m.def("setFrequentOpNameData", &setFrequentOpNameData);
    m.def("updateStep", &updateStep);
    m.def("updateProfiler", &updateProfiler);
    m.def("recordTensorPtrWithTypes", &recordTensorPtrWithTypes, "record tensor type and tensor unique ptr");
    m.def("init_cpp_manager", &InitCppManager, "init cpp manager");
    m.def("deinit_cpp_manager", &DeInitCppManager, "deinit cpp manager");
}
