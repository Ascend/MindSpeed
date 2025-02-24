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
#include "NPUSwapManager.h"

#include <sstream>

#include <ATen/record_function.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/CachingHostAllocator.h>
#include <torch_npu/csrc/core/npu/NPUMemorySwap.h>
#include <torch_npu/csrc/framework/OpHook.h>

#include "swap_log.h"
#include "SwapException.h"

namespace c10_npu {
namespace swap {
SwapStage::SwapStage() : stageType(SwapStageType::INIT), microBatchIndex(0), layerIndex(0) {}

bool SwapStage::operator == (const SwapStage &other) const
{
    return stageType == other.stageType && microBatchIndex == other.microBatchIndex && layerIndex == other.layerIndex;
}

std::ostream &operator << (std::ostream &os, const SwapStage &obj)
{
    os << "SwapStage: "
       << "stageType: " << static_cast<int>(obj.stageType) << " "
       << "microBatchIndex: " << obj.microBatchIndex << " "
       << "layerIndex: " << obj.layerIndex << std::endl;
    return os;
}

SwapConfig::SwapConfig()
    : microBatchNum(0),
      layerNum(0),
      isOOM(false),
      step(0),
      oneStepDuration(0.0),
      policyStep(0),
      enableProfiler(false),
      tensorSizeThresh(0),
      enableExecutor(false),
      enableCustomRecordStream(true)
{}

UniqueSwapPtr::UniqueSwapPtr() : ptrBase(0), index(0) {}

bool UniqueSwapPtr::operator == (const UniqueSwapPtr &other) const
{
    return ptrBase == other.ptrBase && index == other.index;
}

std::ostream &operator << (std::ostream &os, const UniqueSwapPtr &obj)
{
    os << "UniqueSwapPtr: "
       << "ptrBase: " << std::hex << obj.ptrBase << std::dec << " "
       << "index: " << obj.index << std::endl;
    return os;
}

UniqueSwapPtr::operator std::string() const
{
    std::stringstream ss;
    ss << ptrBase << "_" << index;
    return ss.str();
}

UniqueSwapMemory::UniqueSwapMemory() : allocated_bytes(0), reserved_bytes(0), active_bytes(0) {}

UniqueSwapMemory::UniqueSwapMemory(int64_t allocated_bytes, int64_t reserved_bytes, int64_t active_bytes)
    : allocated_bytes(allocated_bytes), reserved_bytes(reserved_bytes), active_bytes(active_bytes)
{}

std::ostream &operator << (std::ostream &os, const UniqueSwapMemory &obj)
{
    os << "UniqueSwapMemory: "
       << "allocated_bytes: " << obj.allocated_bytes << " "
       << "reserved_bytes: " << obj.reserved_bytes << " "
       << "active_bytes: " << obj.active_bytes << std::endl;
    return os;
}

// class ProfilerTensorInfo
ProfilerTensorInfo::ProfilerTensorInfo(const at::Tensor &tensor)
{
    this->ptr = NPUSwapManager::GetInstance().getUniqueSwapPtr(tensor);
    this->nbytes = tensor.storage().nbytes();
    this->dtype = tensor.scalar_type();

    // 根据tensorPtrTypeMap进行查找
    auto tensorPtrTypeIter = NPUSwapManager::GetInstance().tensorPtrTypeMap.find(this->ptr);
    if (tensorPtrTypeIter == NPUSwapManager::GetInstance().tensorPtrTypeMap.end()) {
        this->tensorType = SwapTensorType::OTHERS;
    } else {
        this->tensorType = tensorPtrTypeIter->second;
    }

    for (int i = 0; i < tensor.sizes().size(); i++) {
        this->shapeV2.push_back(tensor.sizes()[i]);
    }
}

std::ostream &operator << (std::ostream &os, const ProfilerTensorInfo &obj)
{
    os << "ProfilerTensorInfo: "
       << "ptr: " << obj.ptr << " "
       << "nbytes: " << obj.nbytes << " "
       << "dtype: " << obj.dtype << " "
       << "tensorType: " << static_cast<int>(obj.tensorType) << " "
       << "shape: " << obj.shapeV2 << std::endl;
    return os;
}

// class ProfilerOpInfo
ProfilerOpInfo::ProfilerOpInfo(int opId, std::string opName, int64_t allocated_bytes, int64_t reserved_bytes,
    int64_t active_bytes)
    : opId(opId), opName(opName), swapMemory(allocated_bytes, reserved_bytes, active_bytes)
{
    this->stage = NPUSwapManager::GetInstance().config.stage;
    this->step = NPUSwapManager::GetInstance().config.step;
}

std::ostream &operator << (std::ostream &os, const ProfilerOpInfo &obj)
{
    os << "ProfilerOpInfo: "
       << "opId: " << obj.opId << " "
       << "opName: " << obj.opName << " "
       << "stage: " << obj.stage << " "
       << "step: " << obj.step << " "
       << "swapMemory: " << obj.swapMemory << std::endl;
    for (auto &t : obj.profilerTensorInfoVec) {
        os << t << std::endl;
    }
    return os;
}

void ProfilerOpInfo::appendTensorInfo(const at::Tensor &tensor)
{
    profilerTensorInfoVec.emplace_back(ProfilerTensorInfo(tensor));
}

ProfilerSwapInfo::ProfilerSwapInfo(int opId, std::string swapName, size_t size, bool isOOM, UniqueSwapPtr srcDataPtr,
    UniqueSwapPtr dstDataPtr)
    : opId(opId), swapName(swapName), size(size), isOOM(isOOM), srcPtr(srcDataPtr), dstPtr(dstDataPtr)
{}

// class SwapProfiler
SwapProfiler::SwapProfiler() : isInit(false) {}

SwapProfiler::~SwapProfiler()
{
    isInit = false;
}

int SwapProfiler::Init()
{
    isInit = true;
    lastOpId = 0;
    return 0;
}

void SwapProfiler::updateStep()
{
    profilerOpInfoMap[NPUSwapManager::GetInstance().config.step] = profilerOpInfoVec;
    lastOpId = profilerOpInfoVec.back().opId;
    profilerOpInfoVec.clear();
    profilerSwapInfoVec.clear();
}

void SwapProfiler::appendOpInfo(std::string &opName, int &opId)
{
    int device = 0;
    SWAP_CHECK_ERROR(c10_npu::GetDevice(&device));
    const c10_npu::NPUCachingAllocator::DeviceStats stats =
        c10_npu::NPUCachingAllocator::allocator.load()->getDeviceStats(device);

    ProfilerOpInfo profilerOpInfo(opId, opName,
        stats.allocated_bytes[static_cast<size_t>(c10_npu::NPUCachingAllocator::StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(c10_npu::NPUCachingAllocator::StatType::AGGREGATE)].current,
        stats.active_bytes[static_cast<size_t>(c10_npu::NPUCachingAllocator::StatType::AGGREGATE)].current);
    profilerOpInfoVec.emplace_back(profilerOpInfo);
}

void SwapProfiler::ReportInfoV2(std::string &opName, int &opId, c10::SmallVector<at::Tensor, N> &tensors)
{
    appendOpInfo(opName, opId);
    ProfilerOpInfo &profilerOpInfo = profilerOpInfoVec.back();
    for (const auto &tensor : tensors) {
        profilerOpInfo.appendTensorInfo(tensor);
    }
}

void SwapProfiler::ReportInfoV2(bool isSwapOut, at::DataPtr &srcDataPtr, at::DataPtr &dstDataPtr, size_t size,
    bool isOOM)
{
    std::string swapName = isSwapOut ? "swapOut" : "swapIn";
    int opId = profilerOpInfoVec.empty() ? lastOpId : profilerOpInfoVec.back().opId;
    ProfilerSwapInfo profilerSwapInfo(opId, swapName, size, isOOM,
        NPUSwapManager::GetInstance().getUniqueSwapPtr(srcDataPtr.get()),
        NPUSwapManager::GetInstance().getUniqueSwapPtr(dstDataPtr.get()));
    profilerSwapInfoVec.emplace_back(profilerSwapInfo);
}

std::vector<ProfilerOpInfo> &SwapProfiler::getPolicyStepOpVec()
{
    return profilerOpInfoMap[NPUSwapManager::GetInstance().config.policyStep];
}

SwapPolicyInfo::SwapPolicyInfo() : executorNeedMatch(false), swapOutOpId(0), swapInOpId(0) {}

std::ostream &operator << (std::ostream &os, const SwapPolicyInfo &obj)
{
    os << "SwapPolicyInfo: "
       << "ptr: " << obj.ptr << " "
       << "swapOutOpId: " << obj.swapOutOpId << " "
       << "swapOutStage: " << obj.swapOutStage << " "
       << "swapInOpId: " << obj.swapInOpId << " "
       << "swapInStage.: " << obj.swapInStage << " "
       << "freeStage: " << obj.freeStage << " "
       << "swapInFreeStage.: " << obj.swapInFreeStage << std::endl;
    return os;
}

ExecutorTensorInfo::ExecutorTensorInfo()
    : opCount(0),
      opTag(0),
      dtype(at::ScalarType::Byte),
      nbytes(0),
      shape(0),
      opCallsStack(0),
      tensorIndexCallsStack(0)
{}

ExecutorTensorInfo::ExecutorTensorInfo(const at::Tensor &tensor, UniqueSwapPtr uniqueSwapPtr)
    : ptr(uniqueSwapPtr),
      opCount(0),
      opTag(0),
      dtype(tensor.scalar_type()),
      opCallsStack(0),
      tensorIndexCallsStack(0)
{
    nbytes = tensor.storage().nbytes();
    shape = convertShapeToInt64(tensor);
}

ExecutorTensorInfo::ExecutorTensorInfo(const SwapStage &s1, const SwapStage &s2)
    : opCount(0),
      opTag(0),
      dtype(at::ScalarType::Byte),
      nbytes(0),
      shape(0),
      opCallsStack(0),
      tensorIndexCallsStack(0),
      swapOutStage(s1),
      swapInStage(s2)
{}

bool ExecutorTensorInfo::operator == (const ExecutorTensorInfo &other) const
{
    return opCount == other.opCount && opTag == other.opTag && dtype == other.dtype && nbytes == other.nbytes &&
        shape == other.shape && opCallsStack == other.opCallsStack &&
        tensorIndexCallsStack == other.tensorIndexCallsStack;
}

std::ostream &operator << (std::ostream &os, const ExecutorTensorInfo &obj)
{
    os << "ExecutorTensorInfo: "
       << "ptr: " << obj.ptr << " "
       << "opCount: " << obj.opCount << " "
       << "opTag: " << obj.opTag << " "
       << "nbytes: " << obj.nbytes << " "
       << "shape: " << obj.shape << " "
       << "opCallsStack: " << obj.opCallsStack << " "
       << "tensorIndexCallsStack: " << obj.tensorIndexCallsStack << " "
       << "swapOutStage: " << obj.swapOutStage << " "
       << "swapInStage: " << obj.swapInStage << " "
       << "freeStage: " << obj.freeStage << " "
       << "swapInFreeStage: " << obj.swapInFreeStage << std::endl;
    return os;
}

size_t ExecutorTensorInfo::convertShapeToInt64(const at::Tensor &tensor)
{
    size_t res = 0;
    for (auto s : tensor.sizes()) {
        res = (res << 16) + s;
    }
    return res;
}

size_t ExecutorTensorInfo::convertShapeToInt64(const c10::SmallVector<size_t, N> &sizes)
{
    size_t res = 0;
    for (auto s : sizes) {
        res = (res << 16) + s;
    }
    return res;
}

void ExecutorTensorInfo::initFromProfilerTensorInfo(const ProfilerTensorInfo &pti)
{
    nbytes = pti.nbytes;
    shape = convertShapeToInt64(pti.shapeV2);
    dtype = pti.dtype;
    ptr.ptrBase = pti.ptr.ptrBase;
    ptr.index = pti.ptr.index;
}

void ExecutorTensorInfo::updateCallsStack(int opOneHot, int opIndex, int tensorIndex)
{
    ++opCount;
    opTag |= opOneHot;
    opCallsStack = (opCallsStack << 8) + opIndex;
    tensorIndexCallsStack = (tensorIndexCallsStack << 8) + tensorIndex;
}

// class SwapExecutor
SwapExecutor::SwapExecutor() : isInit(false) {}

SwapExecutor::~SwapExecutor()
{
    DeInit();
}

int SwapExecutor::Init()
{
    if (isInit) {
        return 0;
    }

    this->swapStreams.push_back(getNPUStreamFromPool(c10_npu::current_device()));

    isInit = true;
    return 0;
}

int SwapExecutor::DeInit()
{
    if (!isInit) {
        return 0;
    }
    isInit = false;
    return 0;
}

int SwapExecutor::SwapOut(c10::intrusive_ptr<c10::StorageImpl> storageImplPtr, bool isOOM, SwapStage *freeStage)
{
    at::DataPtr &dataPtrNpu = storageImplPtr->mutable_data_ptr();
    if (dataPtrNpu.device().is_cpu()) {
        SWAP_LOG_WARN("SwapOut tensor dataPtr is on cpu, skip.");
        return 1;
    }
    uint64_t uniqueId = static_cast<torch_npu::NPUStorageImpl *>(storageImplPtr.get())->get_unique_id();
    auto inEventIter = swapInEventMap.find(uniqueId);
    if (inEventIter != swapInEventMap.end()) {
        SWAP_LOG_WARN("SwapOut tensor need to process swapin wait task, skip.");
        return 1;
    }

    auto swapOutStorageImpIter = swapOutStorageImplMap.find(uniqueId);
    if (swapOutStorageImpIter != swapOutStorageImplMap.end()) {
        SWAP_LOG_WARN("Tensor cannot be swapped out twice consecutively, skip.");
        return 1;
    }

    RECORD_FUNCTION("swap_out", std::vector<c10::IValue>({}));

    SWAP_LOG_INFO("SwapOut pre, storage uniqueId[%lu], mem ptr on npu[%p][%s]", uniqueId, storageImplPtr->data(),
        std::string(NPUSwapManager::GetInstance().getUniqueSwapPtr(storageImplPtr->data())).c_str());

    auto allocatorCPU = at_npu::native::getCachingHostAllocator();
    size_t size = storageImplPtr->nbytes();
    at::DataPtr dataPtrCpu = allocatorCPU->allocate(size);

    NPUSwapManager::GetInstance().tensorPtrCountMap[reinterpret_cast<size_t>(dataPtrCpu.get())]++;

    if (NPUSwapManager::GetInstance().config.enableProfiler) {
        NPUSwapManager::GetInstance().ReportInfoToSwapProfiler(true, dataPtrNpu, dataPtrCpu, size, isOOM);
    }

    c10_npu::NPUStream &swapStream = this->swapStreams[0];
    c10_npu::NPUEvent event;
    event.record(c10_npu::getCurrentNPUStream());
    event.block(swapStream);

    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(swapStream);
    at_npu::native::memory_swap(dataPtrCpu.get(), size, dataPtrNpu.get(), size, 2);
    c10_npu::setCurrentNPUStream(currentStream);

    if (!NPUSwapManager::GetInstance().config.enableCustomRecordStream) {
        c10_npu::NPUCachingAllocator::allocator.load()->recordStream(dataPtrNpu, swapStream);
    }

    dataPtrCpu.unsafe_set_device(dataPtrNpu.device());
    if (NPUSwapManager::GetInstance().config.enableCustomRecordStream) {
        NPUSwapManager::GetInstance().RecordStream(dataPtrNpu, swapStream, freeStage);
    }
    storageImplPtr->set_data_ptr_noswap(std::move(dataPtrCpu));
    SWAP_LOG_INFO("SwapOut post, storage uniqueId[%lu], mem ptr on cpu[%p][%s]", uniqueId, storageImplPtr->data(),
        std::string(NPUSwapManager::GetInstance().getUniqueSwapPtr(storageImplPtr->data())).c_str());

    swapOutStorageImplMap.insert(std::make_pair(uniqueId, c10::weak_intrusive_ptr<c10::StorageImpl>(storageImplPtr)));

    return 0;
}

int SwapExecutor::SwapOut(const at::Tensor &tensor, SwapStage *freeStage)
{
    c10::intrusive_ptr<c10::StorageImpl> storageImplPtr = tensor.storage().getWeakStorageImpl().lock();
    if (!storageImplPtr) {
        return 1;
    }

    return SwapOut(storageImplPtr, false, freeStage);
}

int SwapExecutor::SwapIn(uint64_t uniqueId, bool needWait)
{
    auto outTensorIter = swapOutStorageImplMap.find(uniqueId);
    if (outTensorIter == swapOutStorageImplMap.end()) {
        return 1;
    }

    c10::intrusive_ptr<c10::StorageImpl> storageImplPtr = outTensorIter->second.lock();

    if (!storageImplPtr) {
        SWAP_LOG_INFO(
            "SwapIn pre: StorageImpl of the tensor for current SwapIn is already destructed since the tensor would \
not be used anymore. swapOutStorageImplMap.find(uniqueId[%lu])->second.weak_count[%zu], use_count[%zu]",
            uniqueId, outTensorIter->second.weak_use_count(), outTensorIter->second.use_count());
        swapOutStorageImplMap.erase(outTensorIter);
        return 1;
    }

    RECORD_FUNCTION("swap_in", std::vector<c10::IValue>({}));

    c10_npu::NPUStream &swapStream = this->swapStreams[0];
    c10_npu::NPUEvent beforeSwapInEvent;
    beforeSwapInEvent.record(c10_npu::getCurrentNPUStream());
    beforeSwapInEvent.block(swapStream);

    at::DataPtr &dataPtrCpu = storageImplPtr->mutable_data_ptr();

    SWAP_LOG_INFO("SwapIn pre, storage uniqueId[%lu], mem ptr on cpu[%p][%s]", uniqueId, storageImplPtr->data(),
        std::string(NPUSwapManager::GetInstance().getUniqueSwapPtr(storageImplPtr->data())).c_str());

    auto allocatorNPU = c10_npu::NPUCachingAllocator::allocator.load();
    size_t size = storageImplPtr->nbytes();
    at::DataPtr dataPtrNpu = allocatorNPU->allocate(size);

    if (NPUSwapManager::GetInstance().config.enableProfiler) {
        NPUSwapManager::GetInstance().ReportInfoToSwapProfiler(false, dataPtrCpu, dataPtrNpu, size);
    }

    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(swapStream);
    at_npu::native::memory_swap(dataPtrNpu.get(), size, dataPtrCpu.get(), size, 1);
    c10_npu::setCurrentNPUStream(currentStream);

    if (NPUSwapManager::GetInstance().config.enableCustomRecordStream) {
        auto it = uniqueIdToSwapInFreeStageMap.find(uniqueId);
        if (it != uniqueIdToSwapInFreeStageMap.end()) {
            NPUSwapManager::GetInstance().RecordStream(storageImplPtr, swapStream, &(it->second));
            uniqueIdToSwapInFreeStageMap.erase(it);
        } else {
            NPUSwapManager::GetInstance().RecordStream(storageImplPtr, swapStream);
        }
        NPUSwapManager::GetInstance().RecordStream(dataPtrCpu, swapStream);
    } else {
        c10_npu::NPUCachingAllocator::allocator.load()->recordStream(dataPtrNpu, swapStream);
        at_npu::native::CachingHostAllocator_recordEvent(dataPtrCpu.get(), swapStream);
    }

    storageImplPtr->set_data_ptr_noswap(std::move(dataPtrNpu));

    SWAP_LOG_INFO("SwapIn post, storage uniqueId[%lu], mem ptr on npu[%p][%s]", uniqueId, storageImplPtr->data(),
        std::string(NPUSwapManager::GetInstance().getUniqueSwapPtr(storageImplPtr->data())).c_str());

    c10_npu::NPUEvent afterSwapInEvent;
    afterSwapInEvent.record(swapStream);

    swapOutStorageImplMap.erase(outTensorIter);
    swapInEventMap.insert(std::make_pair(uniqueId, std::move(afterSwapInEvent)));

    if (needWait) {
        SwapInWait(uniqueId);
    }
    return 0;
}

int SwapExecutor::SwapInWait(uint64_t uniqueId)
{
    auto inEventIter = swapInEventMap.find(uniqueId);
    if (inEventIter == swapInEventMap.end()) {
        return 1;
    }

    RECORD_FUNCTION("swap_in_wait", std::vector<c10::IValue>({}));

    SWAP_LOG_INFO("SwapIn wait, storage uniqueId[%lu]", uniqueId);
    inEventIter->second.block(c10_npu::getCurrentNPUStream());
    swapInEventMap.erase(inEventIter);
    return 0;
}

void SwapExecutor::CheckAndInsertStorageToMap(const at::Tensor &src, const at::Tensor &dst)
{
    uint64_t uniqueIdSrc =
        static_cast<torch_npu::NPUStorageImpl *>(src.storage().unsafeGetStorageImpl())->get_unique_id();
    auto tensorIter = swapOutStorageImplMap.find(uniqueIdSrc);
    if (tensorIter == swapOutStorageImplMap.end()) {
        return;
    }

    c10::intrusive_ptr<c10::StorageImpl> storageImplPtrDst = dst.storage().getWeakStorageImpl().lock();
    if (!storageImplPtrDst) {
        return;
    }

    uint64_t uniqueIdDst =
        static_cast<torch_npu::NPUStorageImpl *>(dst.storage().unsafeGetStorageImpl())->get_unique_id();
    swapOutStorageImplMap.insert(
        std::make_pair(uniqueIdDst, c10::weak_intrusive_ptr<c10::StorageImpl>(storageImplPtrDst)));
    SWAP_LOG_INFO("Insert storage to SwapOutStorageImplMap, uniqueId[%lu], mem ptr on cpu[%p][%s]", uniqueIdDst,
        storageImplPtrDst->data(),
        std::string(NPUSwapManager::GetInstance().getUniqueSwapPtr(storageImplPtrDst->data())).c_str());
}

bool SwapExecutor::needGenerateTensorInfo(const at::Tensor &tensor)
{
    if (tensor.nbytes() < NPUSwapManager::GetInstance().config.tensorSizeThresh) {
        return false;
    }
    return true;
}

void SwapExecutor::initOpNameToOneHotAndIndexMap(const std::vector<std::string> &opNames)
{
    opNameToOneHotAndIndexMap.clear();
    size_t oneHot = 1;
    size_t opIndex = 1;
    for (const auto &opName : opNames) {
        opNameToOneHotAndIndexMap[opName] = std::make_pair(oneHot, opIndex);
        oneHot = oneHot << 1;
        opIndex += 1;
    }
}

bool SwapExecutor::checkMatchAndSwapOut(ExecutorTensorInfo &eti, std::vector<ExecutorTensorInfo *> &candidateSwapOutVec)
{
    int matchCount = 0;
    for (auto it = candidateSwapOutVec.rbegin(); matchCount < 5 && it != candidateSwapOutVec.rend(); ++it) {
        ++matchCount;
        if ((*(*it)) == eti && NPUSwapManager::GetInstance().config.stage == (*it)->swapOutStage) {
            eti.swapInStage = (*it)->swapInStage;
            eti.freeStage = (*it)->freeStage;
            eti.swapInFreeStage = (*it)->swapInFreeStage;
            candidateSwapOutVec.erase(std::next(it).base());
            return true;
        }
    }
    return false;
}

void SwapExecutor::initStanderdSwapOutVec(std::vector<ExecutorTensorInfo *> &standerdSwapOutVec,
    const std::vector<ProfilerOpInfo> &opInfosVec, const std::vector<SwapPolicyInfo> &policyInfosVec)
{
    for (const auto &policyInfo : policyInfosVec) {
        if (!policyInfo.executorNeedMatch) {
            continue;
        }
        ExecutorTensorInfo *eti = new ExecutorTensorInfo(policyInfo.swapOutStage, policyInfo.swapInStage);
        for (const auto &opInfoIter : opInfosVec) {
            if (opInfoIter.opId > policyInfo.swapOutOpId) {
                break;
            }
            std::pair<size_t, size_t> oneHotAndIndex = GetOpOneHotAndIndex(opInfoIter.opName);
            int tensorIndex = 0;
            for (const auto &tensorInfoIter : opInfoIter.profilerTensorInfoVec) {
                if (tensorInfoIter.ptr == policyInfo.ptr) {
                    if (eti->opCount == 0) {
                        eti->initFromProfilerTensorInfo(tensorInfoIter);
                        eti->swapInStage = policyInfo.swapInStage;
                        eti->swapOutStage = policyInfo.swapOutStage;
                        eti->freeStage = policyInfo.freeStage;
                        eti->swapInFreeStage = policyInfo.swapInFreeStage;
                    }
                    eti->updateCallsStack(oneHotAndIndex.first, oneHotAndIndex.second, tensorIndex);
                }
                tensorIndex++;
            }
        }
        standerdSwapOutVec.push_back(eti);
    }
}

void SwapExecutor::initCandidateOptimPolicyVec(const std::vector<SwapPolicyInfo> &policyInfosVec)
{
    for (const auto &policyInfo : policyInfosVec) {
        if (policyInfo.executorNeedMatch) {
            continue;
        }
        candidateOptimPolicyVec.emplace_back(policyInfo);
    }
}

void SwapExecutor::processOptimTask(std::unordered_map<UniqueSwapPtr, c10::weak_intrusive_ptr<c10::StorageImpl>,
    HashUniqueSwapPtr> &tensorPtrWeakPtrMap)
{
    for (const auto &policyInfo : candidateOptimPolicyVec) {
        auto weakPtr = tensorPtrWeakPtrMap.find(policyInfo.ptr);
        if (weakPtr != tensorPtrWeakPtrMap.end()) {
            auto storageImplPtr = weakPtr->second.lock();
            if (!storageImplPtr) {
                continue;
            }
            // swapout
            auto tensorToSwapOutVecIter = stageToSwapOutMap
                                              .try_emplace(policyInfo.swapOutStage,
                c10::SmallVector<c10::weak_intrusive_ptr<c10::StorageImpl>, N>())
                                              .first;
            tensorToSwapOutVecIter->second.push_back(weakPtr->second);
            // swapin
            auto tensorToSwapInVecIter = stageToSwapInMap
                                             .try_emplace(policyInfo.swapInStage,
                c10::SmallVector<c10::weak_intrusive_ptr<c10::StorageImpl>, N>())
                                             .first;
            tensorToSwapInVecIter->second.push_back(weakPtr->second);

            auto iter = stageToOptimFreeStageMap.try_emplace(policyInfo.swapOutStage, std::vector<SwapStage>()).first;
            iter->second.push_back(policyInfo.freeStage);

            uint64_t uniqueId =
                static_cast<torch_npu::NPUStorageImpl *>(storageImplPtr.get())->get_unique_id();
            uniqueIdToSwapInFreeStageMap[uniqueId] = policyInfo.swapInFreeStage;
        }
    }
}

std::pair<size_t, size_t> SwapExecutor::GetOpOneHotAndIndex(const std::string &opName)
{
    auto it = opNameToOneHotAndIndexMap.find(opName);
    if (it != opNameToOneHotAndIndexMap.end()) {
        return it->second;
    }
    return std::pair<size_t, size_t>(0, 0);
}

void SwapExecutor::ProcessTensorMatchTask(const std::string &opName, const c10::SmallVector<at::Tensor, N> &curTensors)
{
    if (candidateSwapOutVec.empty()) {
        return;
    }
    std::pair<size_t, size_t> oneHotAndIndex = GetOpOneHotAndIndex(opName);
    int tensorIndex = 0;
    for (const auto &tensor : curTensors) {
        if (needGenerateTensorInfo(tensor)) {
            UniqueSwapPtr uniqueSwapPtr = NPUSwapManager::GetInstance().getUniqueSwapPtr(tensor);
            auto executorTensorInfoIter = ptrToTensorInfoMap.find(uniqueSwapPtr);
            if (executorTensorInfoIter == ptrToTensorInfoMap.end()) {
                executorTensorInfoIter =
                    ptrToTensorInfoMap.try_emplace(uniqueSwapPtr, ExecutorTensorInfo(tensor, uniqueSwapPtr)).first;
            }
            (executorTensorInfoIter->second).updateCallsStack(oneHotAndIndex.first, oneHotAndIndex.second, tensorIndex);
            if (checkMatchAndSwapOut(executorTensorInfoIter->second, candidateSwapOutVec)) {
                SwapOut(tensor, &(executorTensorInfoIter->second.freeStage));
                auto tensorToSwapInVecIter = stageToSwapInMap
                                                 .try_emplace((executorTensorInfoIter->second).swapInStage,
                    c10::SmallVector<c10::weak_intrusive_ptr<c10::StorageImpl>, N>())
                                                 .first;
                tensorToSwapInVecIter->second.push_back(tensor.storage().getWeakStorageImpl());

                uint64_t uniqueId = static_cast<torch_npu::NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl())
                    ->get_unique_id();
                uniqueIdToSwapInFreeStageMap[uniqueId] = executorTensorInfoIter->second.swapInFreeStage;
            }
        }
        tensorIndex++;
    }
}

void SwapExecutor::ProcessStageMatchTask(const SwapStage &currentStage)
{
    auto itOut = stageToSwapOutMap.find(currentStage);
    if (itOut != stageToSwapOutMap.end()) {
        auto tempIter = stageToOptimFreeStageMap.find(currentStage);
        int count = 0;
        for (auto &storageImpl : itOut->second) {
            auto storageImplPtr = storageImpl.lock();
            if (!storageImplPtr) {
                count++;
                continue;
            }
            SwapOut(storageImplPtr, false, &(tempIter->second[count++]));
        }
        stageToSwapOutMap.erase(itOut);
        stageToOptimFreeStageMap.erase(tempIter);
    }

    auto itIn = stageToSwapInMap.find(currentStage);
    if (itIn != stageToSwapInMap.end()) {
        for (auto storageImpl = itIn->second.rbegin(); storageImpl != itIn->second.rend(); ++storageImpl) {
            SwapIn(*storageImpl);
        }
        stageToSwapInMap.erase(itIn);
    }
}

void SwapExecutor::clearStanderdSwapOutVec()
{
    for (auto it = standerdSwapOutVec.begin(); it != standerdSwapOutVec.end(); ++it) {
        delete *it;
    }
    standerdSwapOutVec.clear();
}

void SwapExecutor::clearCandidateOptimPolicyVec()
{
    candidateOptimPolicyVec.clear();
}

void SwapExecutor::SwapIn(c10::weak_intrusive_ptr<c10::StorageImpl> &storageImplWeakPtr)
{
    auto storageImplPtr = storageImplWeakPtr.lock();
    if (!storageImplPtr) {
        return;
    }
    uint64_t uniqueId = static_cast<torch_npu::NPUStorageImpl *>(storageImplPtr.get())->get_unique_id();
    SwapIn(uniqueId, false);
}

void SwapExecutor::SwapOut(c10::weak_intrusive_ptr<c10::StorageImpl> &storageImplWeakPtr)
{
    auto storageImplPtr = storageImplWeakPtr.lock();
    if (!storageImplPtr) {
        return;
    }
    SwapOut(storageImplPtr);
}

void SwapExecutor::updateStep(std::unordered_map<UniqueSwapPtr, c10::weak_intrusive_ptr<c10::StorageImpl>,
    HashUniqueSwapPtr> &tensorPtrWeakPtrMap)
{
    ptrToTensorInfoMap.clear();
    candidateSwapOutVec.clear();
    candidateSwapOutVec.resize(standerdSwapOutVec.size());
    std::reverse_copy(standerdSwapOutVec.begin(), standerdSwapOutVec.end(), candidateSwapOutVec.begin());
    processOptimTask(tensorPtrWeakPtrMap);
}

template <class T> RecordStreamManager<T>::RecordStreamManager() : isInit(false) {}

template <class T> RecordStreamManager<T>::~RecordStreamManager()
{
    isInit = false;
}

template <class T> int RecordStreamManager<T>::Init()
{
    if (isInit) {
        return 0;
    }
    isInit = true;
    return 0;
}

template <class T> int RecordStreamManager<T>::DeInit()
{
    isInit = false;
    return 0;
}

template <class T> void RecordStreamManager<T>::RecordStream(T &ptr, c10_npu::NPUStream stream)
{
    if (!isInit) {
        return;
    }
    c10_npu::NPUEvent recordStreamEvent;
    recordStreamEvent.record(stream);
    recordedQueue.push_back(std::make_pair(std::move(ptr), std::move(recordStreamEvent)));
}

template <class T> void RecordStreamManager<T>::ProcessEvent()
{
    if (!isInit) {
        return;
    }
    while (!recordedQueue.empty()) {
        auto &recordStreamEvent = recordedQueue.front().second;

        if (recordStreamEvent.query()) {
            recordedQueue.pop_front();
        } else {
            break;
        }
    }
}

template <class T> bool RecordStreamManager<T>::ProcessMallocEvent()
{
    if (!isInit) {
        return false;
    }
    bool res = false;
    while (!recordedQueue.empty()) {
        auto &recordStreamEvent = recordedQueue.front().second;
        recordStreamEvent.block(c10_npu::getCurrentNPUStream());
        recordedQueue.pop_front();
        res = true;
    }
    return res;
}

template <class T> RecordStreamWithFreeStageManager<T>::RecordStreamWithFreeStageManager() : isInit(false) {}

template <class T> RecordStreamWithFreeStageManager<T>::~RecordStreamWithFreeStageManager()
{
    isInit = false;
}

template <class T> int RecordStreamWithFreeStageManager<T>::Init()
{
    if (isInit) {
        return 0;
    }
    isInit = true;
    return 0;
}

template <class T> int RecordStreamWithFreeStageManager<T>::DeInit()
{
    isInit = false;
    return 0;
}

template <class T>
void RecordStreamWithFreeStageManager<T>::RecordStream(T &ptr, c10_npu::NPUStream stream, SwapStage &freeStage)
{
    if (!isInit) {
        return;
    }
    c10_npu::NPUEvent recordStreamEvent;
    recordStreamEvent.record(stream);
    auto stageToFreeIter =
        StageToFreeEventMap.try_emplace(freeStage, std::deque<std::pair<T, c10_npu::NPUEvent>>()).first;
    stageToFreeIter->second.push_back(std::make_pair(std::move(ptr), std::move(recordStreamEvent)));
}

template <class T> void RecordStreamWithFreeStageManager<T>::ProcessEvent()
{
    if (!isInit) {
        return;
    }
    for (const auto &pair : StageToFreeEventMap) {
        const SwapStage &stage = pair.first;
        const std::pair<T, c10_npu::NPUEvent> &recordedQueue = pair.second;
        while (!recordedQueue.empty()) {
            auto &recordStreamEvent = recordedQueue.front().second;
            if (recordStreamEvent.query()) {
                recordedQueue.pop_front();
            } else {
                break;
            }
        }
    }
}

template <class T> bool RecordStreamWithFreeStageManager<T>::FreeEventWithStage(SwapStage &freeStage)
{
    if (!isInit) {
        return false;
    }
    bool res = false;
    auto stageToFreeIter = StageToFreeEventMap.find(freeStage);
    if (stageToFreeIter == StageToFreeEventMap.end()) {
        return false;
    }
    auto &recordedQueue = stageToFreeIter->second;
    while (!recordedQueue.empty()) {
        auto &recordStreamEvent = recordedQueue.front().second;
        recordStreamEvent.block(c10_npu::getCurrentNPUStream());
        recordedQueue.pop_front();
        res = true;
    }
    return res;
}

template <class T> bool RecordStreamWithFreeStageManager<T>::ProcessMallocEvent()
{
    if (!isInit) {
        return false;
    }
    bool res = false;

    for (auto &pair : StageToFreeEventMap) {
        const SwapStage &stage = pair.first;
        auto &recordedQueue = pair.second;
        while (!recordedQueue.empty()) {
            auto &recordStreamEvent = recordedQueue.front().second;
            recordStreamEvent.block(c10_npu::getCurrentNPUStream());
            recordedQueue.pop_front();
            res = true;
        }
    }
    return res;
}

// class NPUSwapManager
NPUSwapManager::NPUSwapManager()
    : swap_enable(false),
      swap_oom_enable(false),
      isInit(false),
      executor(nullptr),
      profiler(nullptr),
      opId(0),
      recordedDataPtrManager(nullptr),
      recordedStorageImplManager(nullptr),
      recordedDataPtrWithFreeStageManager(nullptr),
      recordedStorageImplWithFreeStageManager(nullptr)
{}

NPUSwapManager::~NPUSwapManager()
{
    DeInit();
}

NPUSwapManager &NPUSwapManager::GetInstance()
{
    static NPUSwapManager instance;
    return instance;
}

int NPUSwapManager::Init()
{
    if (isInit) {
        return 0;
    }
    if (executor == nullptr) {
        executor = new SwapExecutor();
        if (executor != nullptr) {
            executor->Init();
        }
    }
    if (profiler == nullptr) {
        profiler = new SwapProfiler();
        if (profiler != nullptr) {
            profiler->Init();
        }
    }
    if (recordedDataPtrManager == nullptr) {
        recordedDataPtrManager = new RecordStreamManager<at::DataPtr>();
        if (recordedDataPtrManager != nullptr) {
            recordedDataPtrManager->Init();
        }
    }
    if (recordedStorageImplManager == nullptr) {
        recordedStorageImplManager = new RecordStreamManager<c10::intrusive_ptr<c10::StorageImpl>>();
        if (recordedStorageImplManager != nullptr) {
            recordedStorageImplManager->Init();
        }
    }
    if (recordedDataPtrWithFreeStageManager == nullptr) {
        recordedDataPtrWithFreeStageManager = new RecordStreamWithFreeStageManager<at::DataPtr>();
        if (recordedDataPtrWithFreeStageManager != nullptr) {
            recordedDataPtrWithFreeStageManager->Init();
        }
    }
    if (recordedStorageImplWithFreeStageManager == nullptr) {
        recordedStorageImplWithFreeStageManager =
            new RecordStreamWithFreeStageManager<c10::intrusive_ptr<c10::StorageImpl>>();
        if (recordedStorageImplWithFreeStageManager != nullptr) {
            recordedStorageImplWithFreeStageManager->Init();
        }
    }

    at_npu::native::RegisterOpHookBeginFn(
        [](const std::string &op_name) -> void { c10_npu::swap::NPUSwapManager::GetInstance().BeginHook(op_name); });
    at_npu::native::RegisterOpHookEndFn([]() -> void {
        c10_npu::swap::NPUSwapManager::GetInstance().PostHook();
        c10_npu::swap::NPUSwapManager::GetInstance().EndHook();
    });
    at_npu::native::RegisterOpHookPreFn([](const at::Tensor &at_tensor) -> void {
        if (!at_tensor.defined()) {
            return;
        }
        c10_npu::swap::NPUSwapManager::GetInstance().TensorHook(at_tensor);
    });
    at_npu::native::RegisterOpHookPostFn([](const at::Tensor &at_tensor) -> void {
        if (!at_tensor.defined()) {
            return;
        }
        c10_npu::swap::NPUSwapManager::GetInstance().TensorHook(at_tensor);
    });

    isInit = true;
    return 0;
}

int NPUSwapManager::DeInit()
{
    if (!isInit) {
        return 0;
    }
    if (executor != nullptr) {
        delete executor;
        executor = nullptr;
    }
    if (profiler != nullptr) {
        delete profiler;
        profiler = nullptr;
    }
    if (recordedDataPtrManager != nullptr) {
        delete recordedDataPtrManager;
        recordedDataPtrManager = nullptr;
    }
    if (recordedStorageImplManager != nullptr) {
        delete recordedStorageImplManager;
        recordedStorageImplManager = nullptr;
    }
    if (recordedDataPtrWithFreeStageManager != nullptr) {
        delete recordedDataPtrWithFreeStageManager;
        recordedDataPtrWithFreeStageManager = nullptr;
    }
    if (recordedStorageImplWithFreeStageManager != nullptr) {
        delete recordedStorageImplWithFreeStageManager;
        recordedStorageImplWithFreeStageManager = nullptr;
    }
    isInit = false;
    return 0;
}

void NPUSwapManager::RecordStream(at::DataPtr &dataPtr, c10_npu::NPUStream stream, SwapStage *freeStage)
{
    if (!isInit) {
        return;
    }
    if (freeStage == nullptr) {
        recordedDataPtrManager->RecordStream(dataPtr, stream);
    } else {
        recordedDataPtrWithFreeStageManager->RecordStream(dataPtr, stream, *freeStage);
    }
}

void NPUSwapManager::RecordStream(c10::intrusive_ptr<c10::StorageImpl> storageImpl, c10_npu::NPUStream stream,
    SwapStage *freeStage)
{
    if (!isInit) {
        return;
    }
    if (freeStage == nullptr) {
        recordedStorageImplManager->RecordStream(storageImpl, stream);
    } else {
        recordedStorageImplWithFreeStageManager->RecordStream(storageImpl, stream, *freeStage);
    }
}

void NPUSwapManager::ProcessEvent()
{
    if (!isInit) {
        return;
    }
    recordedDataPtrManager->ProcessEvent();
    recordedStorageImplManager->ProcessEvent();
}

bool NPUSwapManager::ProcessMallocEvent()
{
    if (!isInit) {
        return false;
    }
    if (!config.enableCustomRecordStream) {
        return false;
    }
    bool res = recordedDataPtrManager->ProcessMallocEvent();
    res = res || recordedDataPtrWithFreeStageManager->ProcessMallocEvent();
    res = res || recordedStorageImplWithFreeStageManager->ProcessMallocEvent();
    return res;
}

int NPUSwapManager::BeginHook(const std::string &opName)
{
    if (!isInit) {
        return 0;
    }

    SWAP_LOG_INFO("BeginHook in, opIdStk.size[%zu], opNameStk.size[%zu], curTensorsStk.size[%zu]", opIdStk.size(),
        curOpNameStk.size(), curTensorsStk.size());

    opIdStk.push_back(opId);
    opId++;
    curOpNameStk.push_back(opName);
    c10::SmallVector<at::Tensor, N> curTensors;
    curTensorsStk.push_back(curTensors);

    ProcessEvent();

    SWAP_LOG_INFO("BeginHook out, opId[%d], opName[%s], curTensors num[%zu]", opIdStk.back(),
        curOpNameStk.back().c_str(), curTensorsStk.back().size());

    return 0;
}

int NPUSwapManager::EndHook()
{
    if (!isInit) {
        return 0;
    }
    SWAP_LOG_INFO("EndHook in, opId[%d], opName[%s], curTensors num[%zu]", opIdStk.back(), curOpNameStk.back().c_str(),
        curTensorsStk.back().size());

    for (auto &tensor : curTensorsStk.back()) {
        SaveTensor(tensor);
    }
    tensorValidMap.clear();

    for (size_t i = 0; i < curTensorsStk.back().size(); ++i) {
        SWAP_LOG_DEBUG(
            "EndHook post, opId[%d], opName[%s], curTensors num[%zu], idx[%zu], storage uniqueId[%lu], mem ptr[%p][%s]",
            opIdStk.back(), curOpNameStk.back().c_str(), curTensorsStk.back().size(), i,
            static_cast<torch_npu::NPUStorageImpl *>(curTensorsStk.back()[i].storage().unsafeGetStorageImpl())
                ->get_unique_id(),
            curTensorsStk.back()[i].storage().data(),
            std::string(getUniqueSwapPtr(curTensorsStk.back()[i])).c_str());
    }

    opIdStk.pop_front();
    curOpNameStk.pop_back();
    curTensorsStk.pop_back();
    SWAP_LOG_INFO("EndHook out, opIdStk.size[%zu], opNameStk.size[%zu], curTensorsStk.size[%zu]", opIdStk.size(),
        curOpNameStk.size(), curTensorsStk.size());

    return 0;
}

int NPUSwapManager::TensorHook(const at::Tensor &tensor)
{
    if (!isInit) {
        return 0;
    }

    if (!tensor.device().is_privateuseone()) {
        return 1;
    }

    uint64_t uniqueId = static_cast<torch_npu::NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl())
        ->get_unique_id();

    SWAP_LOG_INFO("TensorHook in, before process, opId[%d], opName[%s], curTensors num[%zu], storage uniqueId[%lu], "
                  "mem ptr[%p][%s]",
        opIdStk.back(), curOpNameStk.back().c_str(), curTensorsStk.back().size(), uniqueId, tensor.storage().data(),
        std::string(getUniqueSwapPtr(tensor)).c_str());

    curTensorsStk.back().emplace_back(tensor);
    tensorValidMap[tensor.storage().mutable_data()] = true;

    executor->SwapInWait(uniqueId);
    executor->SwapIn(uniqueId, true);

    SWAP_LOG_INFO("TensorHook out, after process, opId[%d], opName[%s], curTensors num[%zu], storage uniqueId[%lu], "
                  "mem ptr[%p][%s]",
        opIdStk.back(), curOpNameStk.back().c_str(), curTensorsStk.back().size(), uniqueId, tensor.storage().data(),
        std::string(getUniqueSwapPtr(tensor)).c_str());

    return 0;
}

int NPUSwapManager::PostHook()
{
    if (!isInit) {
        return 0;
    }

    SWAP_LOG_INFO("PostHook in, opId[%d], opName[%s], curTensors num[%zu]", opIdStk.back(), curOpNameStk.back().c_str(),
        curTensorsStk.back().size());

    for (size_t i = 0; i < curTensorsStk.back().size(); ++i) {
        SWAP_LOG_DEBUG("PostHook before process, opId[%d], opName[%s], curTensors num[%zu], idx[%zu], storage \
uniqueId[%lu], mem ptr[%p][%s]",
            opIdStk.back(), curOpNameStk.back().c_str(), curTensorsStk.back().size(), i,
            static_cast<torch_npu::NPUStorageImpl *>(curTensorsStk.back()[i].storage().unsafeGetStorageImpl())
                ->get_unique_id(),
            curTensorsStk.back()[i].storage().data(),
            std::string(getUniqueSwapPtr(curTensorsStk.back()[i])).c_str());
    }

    if (config.enableProfiler) {
        profiler->ReportInfoV2(curOpNameStk.back(), opIdStk.front(), curTensorsStk.back());
    }

    if (config.enableExecutor) {
        executor->ProcessTensorMatchTask(curOpNameStk.back(), curTensorsStk.back());
        executor->ProcessStageMatchTask(config.stage);
        recordedDataPtrWithFreeStageManager->FreeEventWithStage(config.stage);
        recordedStorageImplWithFreeStageManager->FreeEventWithStage(config.stage);
        UpdateCurrentStagePerOp();
    }

    for (size_t i = 0; i < curTensorsStk.back().size(); ++i) {
        SWAP_LOG_DEBUG("PostHook after process, opId[%d], opName[%s], curTensors num[%zu], idx[%zu], storage \
uniqueId[%lu], mem ptr[%p][%s]",
            opIdStk.back(), curOpNameStk.back().c_str(), curTensorsStk.back().size(), i,
            static_cast<torch_npu::NPUStorageImpl *>(curTensorsStk.back()[i].storage().unsafeGetStorageImpl())
                ->get_unique_id(),
            curTensorsStk.back()[i].storage().data(),
            std::string(getUniqueSwapPtr(curTensorsStk.back()[i])).c_str());
    }
    SWAP_LOG_INFO("PostHook out, opId[%d], opName[%s], curTensors num[%zu]", opIdStk.back(),
        curOpNameStk.back().c_str(), curTensorsStk.back().size());
    return 0;
}

void NPUSwapManager::SaveTensor(const at::Tensor &tensor)
{
    if (!swap_oom_enable) {
        return;
    }

    void *dataPtr = tensor.storage().mutable_data();
    auto storageImplIter = storageImplMap.find(dataPtr);
    if (storageImplIter == storageImplMap.end()) {
        storageImplMap.emplace(dataPtr, tensor.storage().getWeakStorageImpl());
    } else {
        storageImplMap.erase(storageImplIter);
        storageImplMap.emplace(dataPtr, tensor.storage().getWeakStorageImpl());
    }

    auto it =
        std::find_if(tensorQueue.begin(), tensorQueue.end(), [&dataPtr](const void *ptr) { return ptr == dataPtr; });
    if (it != tensorQueue.end()) {
        tensorQueue.erase(it);
    }
    tensorQueue.push_back(dataPtr);
}

void NPUSwapManager::CheckAndSwapOutForOOM(void *ptrInBlock)
{
    if (!swap_oom_enable) {
        return;
    }

    auto storageImplIter = storageImplMap.find(ptrInBlock);
    if (storageImplIter == storageImplMap.end()) {
        return;
    }

    c10::intrusive_ptr<c10::StorageImpl> storageImplPtr = storageImplIter->second.lock();
    if (storageImplPtr) {
        auto validIter = tensorValidMap.find(ptrInBlock);
        if (validIter == tensorValidMap.end()) {
            auto blacklistIter = ptrBlacklist.find(getUniqueSwapPtr(storageImplPtr->mutable_data()));
            if (blacklistIter == ptrBlacklist.end()) {
                executor->SwapOut(storageImplPtr, true);
                c10_npu::NPUStream &swapStream = executor->swapStreams[0];
                swapStream.synchronize();
            }
        }
    }
    storageImplMap.erase(storageImplIter);

    auto it = std::find_if(tensorQueue.begin(), tensorQueue.end(),
        [&ptrInBlock](const void *ptr) { return ptr == ptrInBlock; });
    if (it != tensorQueue.end()) {
        tensorQueue.erase(it);
    }
}

std::map<void *, c10::weak_intrusive_ptr<c10::StorageImpl>> &NPUSwapManager::GetStorageImplMap()
{
    return storageImplMap;
}

std::deque<void *> &NPUSwapManager::GetTensorQueue()
{
    return tensorQueue;
}

void NPUSwapManager::ReportInfoToSwapProfiler(bool isSwapOut, at::DataPtr &srcDataPtr, at::DataPtr &dstDataPtr,
    size_t size, bool isOOM)
{
    if (!isInit) {
        return;
    }
    profiler->ReportInfoV2(isSwapOut, srcDataPtr, dstDataPtr, size, isOOM);
}

void NPUSwapManager::CheckAndInsertStorageToMap(const at::Tensor &src, const at::Tensor &dst)
{
    if (!isInit) {
        return;
    }
    executor->CheckAndInsertStorageToMap(src, dst);
}

UniqueSwapPtr NPUSwapManager::getUniqueSwapPtr(const at::Tensor &tensor)
{
    size_t ptrBase = reinterpret_cast<size_t>(tensor.storage().data());
    UniqueSwapPtr uniqueSwapPtr;
    uniqueSwapPtr.ptrBase = ptrBase;
    auto it = tensorPtrCountMap.find(ptrBase);
    if (it == tensorPtrCountMap.end()) {
        uniqueSwapPtr.index = 0;
    } else {
        uniqueSwapPtr.index = tensorPtrCountMap[ptrBase];
    }
    return uniqueSwapPtr;
}

UniqueSwapPtr NPUSwapManager::getUniqueSwapPtr(const void *storagePtr)
{
    size_t ptrBase = reinterpret_cast<size_t>(storagePtr);
    UniqueSwapPtr uniqueSwapPtr;
    uniqueSwapPtr.ptrBase = ptrBase;
    auto it = tensorPtrCountMap.find(ptrBase);
    if (it == tensorPtrCountMap.end()) {
        uniqueSwapPtr.index = 0;
    } else {
        uniqueSwapPtr.index = tensorPtrCountMap[ptrBase];
    }
    return uniqueSwapPtr;
}

UniqueSwapPtr NPUSwapManager::getUniqueSwapPtr(size_t p)
{
    size_t ptrBase = p;
    UniqueSwapPtr uniqueSwapPtr;
    uniqueSwapPtr.ptrBase = ptrBase;
    auto it = tensorPtrCountMap.find(ptrBase);
    if (it == tensorPtrCountMap.end()) {
        uniqueSwapPtr.index = 0;
    } else {
        uniqueSwapPtr.index = tensorPtrCountMap[ptrBase];
    }
    return uniqueSwapPtr;
}

std::vector<UniqueSwapPtr> NPUSwapManager::recordTensorPtrWithTypes(const std::vector<at::Tensor> &tensors,
    SwapTensorType type, int updateWeakPtrMap, bool isUpdateBlacklist)
{
    if (updateWeakPtrMap == 1) {
        tensorPtrWeakPtrMap.clear();
    }

    std::vector<UniqueSwapPtr> results;
    results.reserve(tensors.size());

    for (const auto &tensor : tensors) {
        auto uniquePtr = getUniqueSwapPtr(tensor);

        tensorPtrTypeMap.try_emplace(uniquePtr, type);

        if (updateWeakPtrMap > 0) {
            tensorPtrWeakPtrMap.try_emplace(uniquePtr, tensor.storage().getWeakStorageImpl());
        }
        if (isUpdateBlacklist) {
            ptrBlacklist.insert(uniquePtr);
        }

        results.emplace_back(uniquePtr);
    }
    return results;
}

void NPUSwapManager::initOpNameToOneHotAndIndexMap(std::vector<std::string> &frequentOpNames)
{
    executor->initOpNameToOneHotAndIndexMap(frequentOpNames);
}

void NPUSwapManager::FunAfterProfiler(std::vector<SwapPolicyInfo> &policyInfoVec)
{
    if (!isInit) {
        return;
    }
    if (config.enableExecutor) {
        executor->clearStanderdSwapOutVec();
        executor->initStanderdSwapOutVec(executor->standerdSwapOutVec, profiler->getPolicyStepOpVec(), policyInfoVec);
        executor->clearCandidateOptimPolicyVec();
        executor->initCandidateOptimPolicyVec(policyInfoVec);
    }
}

void NPUSwapManager::UpdateCurrentStagePerOp()
{
    if (config.fwdOpLayerInfo.empty() || config.bwdOpLayerInfo.empty()) {
        return;
    }
    config.currentStageOpId++;
    if (config.stage.stageType == SwapStageType::FWD) {
        for (int i = 0; i < config.fwdOpLayerInfo.size(); i++) {
            if (config.currentStageOpId <= config.fwdOpLayerInfo[i]) {
                config.stage.layerIndex = i + 1;
                break;
            }
        }
        if (config.currentStageOpId > config.fwdOpLayerInfo.back()) {
            config.stage.layerIndex = config.fwdOpLayerInfo.size() + 1;
        }
    } else if (config.stage.stageType == SwapStageType::BWD) {
        for (int i = 0; i < config.bwdOpLayerInfo.size(); i++) {
            if (config.currentStageOpId <= config.bwdOpLayerInfo[i]) {
                config.stage.layerIndex = i + 1;
                break;
            }
        }
        if (config.currentStageOpId > config.bwdOpLayerInfo.back()) {
            config.stage.layerIndex = config.bwdOpLayerInfo.size() + 1;
        }
    }
}

void NPUSwapManager::updateStep()
{
    if (!isInit) {
        return;
    }
    config.currentStageOpId = 0;
    executor->updateStep(tensorPtrWeakPtrMap);
    tensorQueue.clear();
    config.isOOM = false;
}

c10_npu::NPUStream &NPUSwapManager::GetSwapStream()
{
    return this->executor->swapStreams[0];
}
} // namespace swap
} // namespace c10_npu
