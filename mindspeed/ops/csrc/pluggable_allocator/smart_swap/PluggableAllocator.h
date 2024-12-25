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

#include "DeviceCachingAllocator.h"

class PluggableAllocator {
private:
    std::mutex mutex;

    // allocated blocks by device pointer
    ska::flat_hash_map<void *, Block *> allocated_blocks;

    mutable std::mutex npu_free_mutex;

    PluggableAllocator() {}

public:
    PluggableAllocator(const PluggableAllocator &) = delete;
    PluggableAllocator &operator = (const PluggableAllocator &) = delete;

    static PluggableAllocator &getInstance()
    {
        static PluggableAllocator instance;
        return instance;
    }

    std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

    std::mutex *getFreeMutex() const;
    void add_allocated_block(Block *block);
    Block *get_allocated_block(void *ptr, bool remove = false);
    void init(int device_count);
    bool initialized();
    void *malloc(int device, size_t size, aclrtStream stream);
    void free(void *ptr);
    void setMemoryFraction(double fraction, int device);
    void emptyCache(bool check_error);
    void recordStream(void *ptr, c10_npu::NPUStream stream);
    void eraseStream(void *ptr, c10_npu::NPUStream stream);
    std::vector<SegmentInfo> snapshot();
    c10::DataPtr allocate(size_t size) const;
    c10::DeleterFnPtr raw_deleter() const;
    void cacheInfo(int dev_id, size_t *cachedAndFree, size_t *largestBlock);
    void assertValidDevice(int device);
    DeviceStats getDeviceStats(int device);
    void resetAccumulatedStats(int device);
    void resetPeakStats(int device);
    void *raw_alloc(size_t nbytes);
    void *raw_alloc_with_stream(size_t nbytes, aclrtStream stream);
    void raw_delete(void *ptr);
    void FreeDeviceCachedMemory(int device);
    std::string name();
};
