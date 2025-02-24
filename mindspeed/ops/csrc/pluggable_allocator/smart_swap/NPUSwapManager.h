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
#pragma once

#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/SmallVector.h>
#include <c10/util/intrusive_ptr.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>

namespace c10_npu {
namespace swap {
enum class TORCH_NPU_API SwapTensorType {
    MODEL,
    OPTIM,
    SHARED_MEMORY,
    OTHERS,
    RESERVED,
};

enum class TORCH_NPU_API SwapStageType {
    INIT = 1,
    FWD,
    BWD,
    OPTIM,
    RESERVED,
};

class TORCH_NPU_API SwapStage {
public:
    SwapStage();

    bool operator == (const SwapStage &other) const;
    friend std::ostream &operator << (std::ostream &os, const SwapStage &obj);

    SwapStageType stageType;
    uint32_t microBatchIndex;
    uint32_t layerIndex;
};

class HashSwapStage {
public:
    size_t operator () (const SwapStage &s) const
    {
        return std::hash<int>()((s.layerIndex) + ((s.microBatchIndex) << 8) + ((static_cast<int>(s.stageType) << 16)));
    }
};

class TORCH_NPU_API SwapConfig {
public:
    SwapConfig();

    // static
    // model config
    uint32_t microBatchNum;
    uint32_t layerNum;

    // update
    bool isOOM;
    SwapStage stage;
    uint32_t step;
    float oneStepDuration;
    uint32_t policyStep;
    int currentStageOpId;

    // update policy
    bool enableProfiler;

    uint64_t tensorSizeThresh;
    bool enableExecutor;
    bool enableCustomRecordStream;
    std::vector<uint32_t> fwdOpLayerInfo;
    std::vector<uint32_t> bwdOpLayerInfo;
};

class TORCH_NPU_API UniqueSwapPtr {
public:
    UniqueSwapPtr();

    bool operator == (const UniqueSwapPtr &other) const;
    bool operator < (const UniqueSwapPtr &other) const
    {
        return ptrBase < other.ptrBase;
    };
    friend std::ostream &operator << (std::ostream &os, const UniqueSwapPtr &obj);
    operator std::string() const;

    size_t ptrBase;
    size_t index;
};

class HashUniqueSwapPtr {
public:
    size_t operator () (const UniqueSwapPtr &p) const
    {
        return std::hash<int>()(p.ptrBase) ^ std::hash<int>()(p.index);
    }
};

class TORCH_NPU_API UniqueSwapMemory {
public:
    UniqueSwapMemory();
    UniqueSwapMemory(int64_t allocated_bytes, int64_t reserved_bytes, int64_t active_bytes);

    friend std::ostream &operator << (std::ostream &os, const UniqueSwapMemory &obj);

    int64_t allocated_bytes;
    int64_t reserved_bytes;
    int64_t active_bytes;
};

class TORCH_NPU_API ProfilerTensorInfo {
public:
    explicit ProfilerTensorInfo(const at::Tensor &tensor);

    friend std::ostream &operator << (std::ostream &os, const ProfilerTensorInfo &obj);

    UniqueSwapPtr &getPtr()
    {
        return ptr;
    }
    size_t &getNbytes()
    {
        return nbytes;
    }
    c10::SmallVector<size_t, N> &getShapeV2()
    {
        return shapeV2;
    }
    at::ScalarType &getDtype()
    {
        return dtype;
    }
    SwapTensorType &getTensorType()
    {
        return tensorType;
    }

    UniqueSwapPtr ptr;
    size_t nbytes;
    at::ScalarType dtype;
    SwapTensorType tensorType;
    c10::SmallVector<size_t, N> shapeV2;
};

class TORCH_NPU_API ProfilerOpInfo {
public:
    ProfilerOpInfo(int opId, std::string opName, int64_t allocated_bytes, int64_t reserved_bytes, int64_t active_bytes);

    friend std::ostream &operator << (std::ostream &os, const ProfilerOpInfo &obj);

    int &getOpId()
    {
        return opId;
    }
    std::string &getOpName()
    {
        return opName;
    }
    SwapStage &getStage()
    {
        return stage;
    }
    uint32_t &getStep()
    {
        return step;
    }
    UniqueSwapMemory &getSwapMemory()
    {
        return swapMemory;
    }
    std::vector<ProfilerTensorInfo> &getProfilerTensorInfo()
    {
        return profilerTensorInfoVec;
    }
    void appendTensorInfo(const at::Tensor &tensor);

    int opId;
    std::string opName;
    SwapStage stage;
    uint32_t step;
    UniqueSwapMemory swapMemory;
    std::vector<ProfilerTensorInfo> profilerTensorInfoVec;
};

class TORCH_NPU_API ProfilerSwapInfo {
public:
    ProfilerSwapInfo(int opId, std::string swapName, size_t size, bool isOOM, UniqueSwapPtr srcDataPtr,
        UniqueSwapPtr dstDataPtr);

    int &getOpId()
    {
        return opId;
    }
    std::string &getSwapName()
    {
        return swapName;
    }
    size_t &getSize()
    {
        return size;
    }
    bool &getIsOOM()
    {
        return isOOM;
    }
    UniqueSwapPtr &getSrcPtr()
    {
        return srcPtr;
    }
    UniqueSwapPtr &getDstPtr()
    {
        return dstPtr;
    }

    int opId;
    std::string swapName;
    size_t size;
    bool isOOM;
    UniqueSwapPtr srcPtr;
    UniqueSwapPtr dstPtr;
};

class TORCH_NPU_API SwapProfiler {
public:
    SwapProfiler();
    ~SwapProfiler();
    int Init();

    void updateStep();
    void appendOpInfo(std::string &opName, int &opId);
    void ReportInfoV2(std::string &opName, int &opId, c10::SmallVector<at::Tensor, N> &tensors);
    void ReportInfoV2(bool isSwapOut, at::DataPtr &srcDataPtr, at::DataPtr &dstDataPtr, size_t size, bool isOOM);
    std::vector<ProfilerOpInfo> &getProfilerOpInfoVec()
    {
        return profilerOpInfoVec;
    }
    std::vector<ProfilerSwapInfo> &getProfilerSwapInfoVec()
    {
        return profilerSwapInfoVec;
    }
    std::vector<ProfilerOpInfo> &getPolicyStepOpVec();

    std::map<uint32_t, std::vector<ProfilerOpInfo>> profilerOpInfoMap;

private:
    bool isInit;
    int lastOpId;
    std::vector<ProfilerOpInfo> profilerOpInfoVec;
    std::vector<ProfilerSwapInfo> profilerSwapInfoVec;
};

class TORCH_NPU_API SwapPolicyInfo {
public:
    SwapPolicyInfo();

    friend std::ostream &operator << (std::ostream &os, const SwapPolicyInfo &obj);

    bool executorNeedMatch;
    UniqueSwapPtr ptr;
    int swapOutOpId;
    int swapInOpId;
    SwapStage swapOutStage;
    SwapStage swapInStage;
    SwapStage freeStage;
    SwapStage swapInFreeStage;
};

class ExecutorTensorInfo {
public:
    ExecutorTensorInfo();
    ExecutorTensorInfo(const at::Tensor &tensor, UniqueSwapPtr uniqueSwapPtr);
    ExecutorTensorInfo(const SwapStage &swapOutStage, const SwapStage &swapInStage);

    bool operator == (const ExecutorTensorInfo &other) const;
    friend std::ostream &operator << (std::ostream &os, const ExecutorTensorInfo &obj);

    size_t convertShapeToInt64(const at::Tensor &tensor);
    size_t convertShapeToInt64(const c10::SmallVector<size_t, N> &sizes);
    void initFromProfilerTensorInfo(const ProfilerTensorInfo &pti);
    void updateCallsStack(int opOneHot, int opIndex, int tensorIndex);

    UniqueSwapPtr ptr;
    size_t opCount;
    size_t opTag;
    at::ScalarType dtype;
    size_t nbytes;
    size_t shape;
    size_t opCallsStack;
    size_t tensorIndexCallsStack;
    SwapStage swapOutStage;
    SwapStage swapInStage;
    SwapStage freeStage;
    SwapStage swapInFreeStage;
};

class SwapExecutor {
public:
    SwapExecutor();
    ~SwapExecutor();
    int Init();
    int DeInit();

    int SwapOut(c10::intrusive_ptr<c10::StorageImpl> storageImplPtr, bool isOOM = false,
        SwapStage *freeStage = nullptr);
    int SwapOut(const at::Tensor &tensor, SwapStage *freeStage = nullptr);
    int SwapIn(uint64_t uniqueId, bool needWait);
    int SwapInWait(uint64_t uniqueId);
    void CheckAndInsertStorageToMap(const at::Tensor &src, const at::Tensor &dst);
    void ProcessTensorMatchTask(const std::string &opName, const c10::SmallVector<at::Tensor, N> &curTensors);
    void ProcessStageMatchTask(const SwapStage &currentStage);
    void updateStep(std::unordered_map<UniqueSwapPtr, c10::weak_intrusive_ptr<c10::StorageImpl>, HashUniqueSwapPtr>
                        &tensorPtrWeakPtrMap);

public:
    bool isInit;
    std::vector<c10_npu::NPUStream> swapStreams;

    std::map<uint64_t, c10::weak_intrusive_ptr<c10::StorageImpl>> swapOutStorageImplMap;
    std::map<uint64_t, c10_npu::NPUEvent> swapInEventMap;
    std::vector<ExecutorTensorInfo *> standerdSwapOutVec;
    std::vector<ExecutorTensorInfo *> candidateSwapOutVec;
    std::vector<SwapPolicyInfo> candidateOptimPolicyVec;
    std::unordered_map<std::string, std::pair<size_t, size_t>> opNameToOneHotAndIndexMap;
    std::unordered_map<UniqueSwapPtr, ExecutorTensorInfo, HashUniqueSwapPtr> ptrToTensorInfoMap;
    std::unordered_map<SwapStage, c10::SmallVector<c10::weak_intrusive_ptr<c10::StorageImpl>, N>, HashSwapStage>
        stageToSwapInMap;
    std::unordered_map<SwapStage, c10::SmallVector<c10::weak_intrusive_ptr<c10::StorageImpl>, N>, HashSwapStage>
        stageToSwapOutMap;

    std::unordered_map<SwapStage, std::vector<SwapStage>, HashSwapStage> stageToOptimFreeStageMap;
    std::unordered_map<uint64_t, SwapStage> uniqueIdToSwapInFreeStageMap;

    std::pair<size_t, size_t> GetOpOneHotAndIndex(const std::string &opName);

    bool needGenerateTensorInfo(const at::Tensor &tensor);
    void initOpNameToOneHotAndIndexMap(const std::vector<std::string> &opNames);
    bool checkMatchAndSwapOut(ExecutorTensorInfo &eti, std::vector<ExecutorTensorInfo *> &candidateSwapOutVec);
    void initStanderdSwapOutVec(std::vector<ExecutorTensorInfo *> &swapOutVec,
        const std::vector<ProfilerOpInfo> &opInfosVec, const std::vector<SwapPolicyInfo> &policyInfosVec);
    void clearStanderdSwapOutVec();
    void clearCandidateOptimPolicyVec();
    void checkStageAndSwapIn(const SwapStage &swapStage);
    void SwapIn(c10::weak_intrusive_ptr<c10::StorageImpl> &storageImplPtr);
    void SwapOut(c10::weak_intrusive_ptr<c10::StorageImpl> &storageImplWeakPtr);
    void initCandidateOptimPolicyVec(const std::vector<SwapPolicyInfo> &policyInfosVec);
    void processOptimTask(std::unordered_map<UniqueSwapPtr, c10::weak_intrusive_ptr<c10::StorageImpl>,
        HashUniqueSwapPtr> &tensorPtrWeakPtrMap);
};

template <class T> class RecordStreamManager {
public:
    RecordStreamManager();
    ~RecordStreamManager();
    int Init();
    int DeInit();

    void RecordStream(T &ptr, c10_npu::NPUStream stream);
    void ProcessEvent();
    bool ProcessMallocEvent();

private:
    bool isInit;
    std::deque<std::pair<T, c10_npu::NPUEvent>> recordedQueue;
};

template <class T> class RecordStreamWithFreeStageManager {
public:
    RecordStreamWithFreeStageManager();
    ~RecordStreamWithFreeStageManager();
    int Init();
    int DeInit();

    void ProcessEvent();
    void RecordStream(T &ptr, c10_npu::NPUStream stream, SwapStage &freeStage);
    bool FreeEventWithStage(SwapStage &freeStage);
    bool ProcessMallocEvent();

private:
    bool isInit;
    std::unordered_map<SwapStage, std::deque<std::pair<T, c10_npu::NPUEvent>>, HashSwapStage> StageToFreeEventMap;
};

class TORCH_NPU_API NPUSwapManager {
public:
    static NPUSwapManager &GetInstance();
    ~NPUSwapManager();
    int Init();
    int DeInit();

    int BeginHook(const std::string &opName);
    int EndHook();
    int TensorHook(const at::Tensor &tensor);
    int PostHook();

    void SaveTensor(const at::Tensor &tensor);
    void CheckAndSwapOutForOOM(void *ptrInBlock);
    std::map<void *, c10::weak_intrusive_ptr<c10::StorageImpl>> &GetStorageImplMap();
    std::deque<void *> &GetTensorQueue();

    void ReportInfoToSwapProfiler(bool isSwapOut, at::DataPtr &srcDataPtr, at::DataPtr &dstDataPtr, size_t size,
        bool isOOM = false);
    void CheckAndInsertStorageToMap(const at::Tensor &src, const at::Tensor &dst);

    void RecordStream(at::DataPtr &dataPtr, c10_npu::NPUStream stream, SwapStage *freeStage = nullptr);
    void RecordStream(c10::intrusive_ptr<c10::StorageImpl> storageImpl, c10_npu::NPUStream stream,
        SwapStage *freeStage = nullptr);
    void ProcessEvent();
    bool ProcessMallocEvent();

    void updateStage();
    void FunAfterProfiler(std::vector<SwapPolicyInfo> &policyInfoVec);
    void updateStep();
    void initOpNameToOneHotAndIndexMap(std::vector<std::string> &frequentOpNames);
    std::vector<UniqueSwapPtr> recordTensorPtrWithTypes(const std::vector<at::Tensor> &tensors, SwapTensorType type,
        int updateWeakPtrMap = 0, // 0: do nothing, 1: clear, 2: append
        bool isUpdateBlacklist = false);
    void UpdateCurrentStagePerOp();

    bool swap_enable;
    bool swap_oom_enable;
    SwapConfig config;
    std::map<size_t, int> tensorPtrCountMap;
    std::unordered_map<UniqueSwapPtr, SwapTensorType, HashUniqueSwapPtr> tensorPtrTypeMap;
    std::unordered_map<UniqueSwapPtr, c10::weak_intrusive_ptr<c10::StorageImpl>, HashUniqueSwapPtr> tensorPtrWeakPtrMap;
    std::set<UniqueSwapPtr> ptrBlacklist;

    SwapProfiler *getSwapProfiler()
    {
        return profiler;
    }
    UniqueSwapPtr getUniqueSwapPtr(const at::Tensor &tensor);
    UniqueSwapPtr getUniqueSwapPtr(const void *storagePtr);
    UniqueSwapPtr getUniqueSwapPtr(size_t p);

    c10_npu::NPUStream &GetSwapStream();

private:
    NPUSwapManager();

    bool isInit;
    SwapExecutor *executor;
    SwapProfiler *profiler;

    // always update
    int opId;
    std::map<void *, c10::weak_intrusive_ptr<c10::StorageImpl>> storageImplMap;
    std::deque<void *> tensorQueue;
    std::map<void *, bool> tensorValidMap;

    // use deque to store current variables to deal with nested OpCommand calls
    std::deque<int> opIdStk;
    std::deque<std::string> curOpNameStk;
    std::deque<c10::SmallVector<at::Tensor, N>> curTensorsStk;

    RecordStreamManager<at::DataPtr> *recordedDataPtrManager;
    RecordStreamManager<c10::intrusive_ptr<c10::StorageImpl>> *recordedStorageImplManager;
    RecordStreamWithFreeStageManager<at::DataPtr> *recordedDataPtrWithFreeStageManager;
    RecordStreamWithFreeStageManager<c10::intrusive_ptr<c10::StorageImpl>> *recordedStorageImplWithFreeStageManager;
};

class SwapOutOfMemError : public std::exception {
public:
    SwapOutOfMemError(const std::string &message, void *data) : message(message), data(data) {}
    const char *what() const noexcept override
    {
        return message.c_str();
    }
    void *GetData() const noexcept
    {
        return data;
    }

private:
    std::string message;
    void *data = nullptr;
};
} // namespace swap
} // namespace c10_npu
