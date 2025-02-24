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
#include "PluggableAllocator.h"

#include <c10/util/irange.h>

#include "swap_log.h"
#include "NPUSwapManager.h"

void local_raw_delete(void *ptr)
{
    PluggableAllocator::getInstance().free(ptr);
}

void PluggableAllocator::add_allocated_block(Block *block)
{
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
}

std::mutex *PluggableAllocator::getFreeMutex() const
{
    return &npu_free_mutex;
}

Block *PluggableAllocator::get_allocated_block(void *ptr, bool remove)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
        return nullptr;
    }
    Block *block = it->second;
    if (remove) {
        allocated_blocks.erase(it);
    }
    return block;
}

void PluggableAllocator::init(int device_count)
{
    int size = static_cast<int>(device_allocator.size());
    if (size < device_count) {
        device_allocator.resize(device_count);
        for (const auto i : c10::irange(size, device_count)) {
            device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
        }
    }
}

bool PluggableAllocator::initialized()
{
    return !device_allocator.empty();
}

/* * allocates a block which is safe to use from the provided stream */
void *PluggableAllocator::malloc(int device, size_t size, aclrtStream stream)
{
    void *devPtr = nullptr;
    if (c10_npu::swap::NPUSwapManager::GetInstance().swap_oom_enable) {
        bool isTryMallocExit = false;
        uint32_t tryMallocCount = 0;
        while (!isTryMallocExit) {
            try {
                Block *block = device_allocator[device]->malloc(device, size, stream);
                add_allocated_block(block);
                devPtr = static_cast<void *>(block->ptr);
                if (devPtr != nullptr) {
                    if (tryMallocCount > 0) {
                        SWAP_LOG_WARN("[SwapOomEnable] try malloc count[%u], finally success!", tryMallocCount);
                    }
                    isTryMallocExit = true;
                }
            } catch (const c10_npu::swap::SwapOutOfMemError &err) {
                c10_npu::swap::NPUSwapManager::GetInstance().CheckAndSwapOutForOOM(err.GetData());
            }
            tryMallocCount++;
        }
    } else {
        Block *block = device_allocator[device]->malloc(device, size, stream);
        add_allocated_block(block);
        devPtr = static_cast<void *>(block->ptr);
    }

    return devPtr;
}

void PluggableAllocator::free(void *ptr)
{
    if (!ptr) {
        return;
    }
    Block *block = get_allocated_block(ptr, true);
    if (!block) {
        AT_ERROR("invalid device pointer: ", ptr);
    }
    device_allocator[block->device]->free(block);
}

void PluggableAllocator::setMemoryFraction(double fraction, int device)
{
    TORCH_INTERNAL_ASSERT(0 <= device && device < device_allocator.size(), "Allocator not initialized for device ",
        device, ": did you call init?");
    TORCH_INTERNAL_ASSERT(0 <= fraction && fraction <= 1, "invalid fraction:", fraction, ". Please set within (0, 1).");

    c10_npu::SetDevice(device);

    device_allocator[device]->setMemoryFraction(fraction);
}

void PluggableAllocator::emptyCache(bool check_error)
{
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++)
        device_allocator[i]->emptyCache(check_error);
}

void PluggableAllocator::recordStream(void *ptr, c10_npu::NPUStream stream)
{
    if (!ptr) {
        return;
    }
    Block *block = get_allocated_block(ptr);
    device_allocator[block->device]->recordStream(block, stream);
}

void PluggableAllocator::eraseStream(void *ptr, c10_npu::NPUStream stream)
{
    if (!ptr) {
        return;
    }
    Block *block = get_allocated_block(ptr);
    if (!block) {
        AT_ERROR("invalid device pointer: ", ptr);
    }

    if (block->stream != c10_npu::getCurrentNPUStream(block->device).stream(false)) {
        // If the Stream applying for tensor block different from
        // the stream of submiting event wait task in HCCL synchronize()
        // method, the recordSteam can not be erased.
        // New tensor creation may use the block before HCCL op is complete.
        return;
    }

    device_allocator[block->device]->eraseStream(block, stream);
}

std::vector<SegmentInfo> PluggableAllocator::snapshot()
{
    std::vector<SegmentInfo> result;
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++) {
        auto snap = device_allocator[i]->snapshot();
        result.insert(result.end(), snap.begin(), snap.end());
    }
    return result;
}

c10::DeleterFnPtr PluggableAllocator::raw_deleter() const
{
    return &local_raw_delete;
}

void PluggableAllocator::cacheInfo(int dev_id, size_t *cachedAndFree, size_t *largestBlock)
{
    device_allocator[dev_id]->cacheInfo(cachedAndFree, largestBlock);
}

void PluggableAllocator::assertValidDevice(int device)
{
    int device_num = c10_npu::device_count();
    AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats PluggableAllocator::getDeviceStats(int device)
{
    assertValidDevice(device);
    return device_allocator[device]->getStats();
}

void PluggableAllocator::resetAccumulatedStats(int device)
{
    assertValidDevice(device);
    device_allocator[device]->resetAccumulatedStats();
}

void PluggableAllocator::resetPeakStats(int device)
{
    assertValidDevice(device);
    device_allocator[device]->resetPeakStats();
}

void PluggableAllocator::raw_delete(void *ptr)
{
    this->free(ptr);
}

void PluggableAllocator::FreeDeviceCachedMemory(int device)
{
    device_allocator[device]->emptyCache(true);
}

std::string PluggableAllocator::name()
{
    return "native";
}
