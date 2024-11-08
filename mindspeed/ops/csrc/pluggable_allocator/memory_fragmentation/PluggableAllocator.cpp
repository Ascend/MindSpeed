#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <vector>

#include <c10/util/irange.h>

#include "PluggableAllocator.h"

void local_raw_delete(void *ptr) {
    PluggableAllocator::getInstance().free(ptr);
}

void PluggableAllocator::add_allocated_block(Block *block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
}

std::mutex *PluggableAllocator::getFreeMutex() const {
    return &npu_free_mutex;
}

Block *PluggableAllocator::get_allocated_block(void *ptr, bool remove) {
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

void PluggableAllocator::init(int device_count) {
    int max_device_count = 1000000;
    TORCH_INTERNAL_ASSERT(device_count < max_device_count, "Error, out of maximum device");
    int size = static_cast<int>(device_allocator.size());
    if (size < device_count) {
        device_allocator.resize(device_count);
        for (const auto i: c10::irange(size, device_count)) {
            device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
        }
    }
}

bool PluggableAllocator::initialized() {
    return !device_allocator.empty();
}

/** allocates a block which is safe to use from the provided stream */
void *PluggableAllocator::malloc(int device, size_t size, aclrtStream stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "device index out of range.");
    Block *block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    void *devPtr = static_cast<void *>(block->ptr);
    return devPtr;
}

void PluggableAllocator::free(void *ptr) {
    if (!ptr) {
        return;
    }
    Block *block = get_allocated_block(ptr, true);
    if (!block) {
        AT_ERROR("invalid device pointer: ", ptr);
    }
    TORCH_INTERNAL_ASSERT(
        0 <= block->device && static_cast<size_t>(block->device) < device_allocator.size(),
        "device index out of range.");
    device_allocator[block->device]->free(block);
}

void PluggableAllocator::setMemoryFraction(double fraction, int device) {
    TORCH_INTERNAL_ASSERT(
            0 <= device && static_cast<size_t>(device) < device_allocator.size(),
            "device index out of range.");
    TORCH_INTERNAL_ASSERT(
            0 <= fraction && fraction <= 1,
            "invalid fraction:",
            fraction,
            ". Please set within (0, 1).");

    TORCH_CHECK(c10_npu::SetDevice(device) == ACL_ERROR_NONE, "SetDevice failed.");

    device_allocator[device]->set_memory_fraction(fraction);
}

void PluggableAllocator::emptyCache(bool check_error) {
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++)
        device_allocator[i]->empty_cache(check_error);
}

void PluggableAllocator::setShutdownStats() {
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++) {
        device_allocator[i]->dev_set_shutdown_stats();
    }
}

void *PluggableAllocator::getBaseAllocation(void *ptr, size_t *outSize) {
    Block *block = get_allocated_block(ptr);
    if (!block) {
        AT_ERROR("invalid device pointer: ", ptr);
    }
    TORCH_INTERNAL_ASSERT(
        0 <= block->device && static_cast<size_t>(block->device) < device_allocator.size(),
        "device index out of range.");
    return device_allocator[block->device]->get_base_allocation(block, outSize);
}

void PluggableAllocator::recordStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
        return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete) {
        return;
    }

    Block *block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    TORCH_INTERNAL_ASSERT(
        0 <= block->device && static_cast<size_t>(block->device) < device_allocator.size(),
        "device index out of range.");
    device_allocator[block->device]->record_stream(block, stream);
}

void PluggableAllocator::eraseStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream) {
    if (!ptr.get()) {
        return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete) {
        return;
    }

    Block *block = get_allocated_block(ptr.get());
    if (!block) {
        AT_ERROR("invalid device pointer: ", ptr.get());
    }

    if (block->stream != c10_npu::getCurrentNPUStream(block->device)) {
        // If the Stream applying for tensor block different from
        // the stream of submiting event wait task in HCCL synchronize()
        // method, the recordSteam can not be erased.
        // New tensor creation may use the block before HCCL op is complete.
        return;
    }

    TORCH_INTERNAL_ASSERT(
        0 <= block->device && static_cast<size_t>(block->device) < device_allocator.size(),
        "device index out of range.");
    device_allocator[block->device]->erase_stream(block, stream);
}

std::vector<SegmentInfo> PluggableAllocator::snapshot() {
    std::vector<SegmentInfo> result;
    int count = static_cast<int>(device_allocator.size());
    for (int i = 0; i < count; i++) {
        auto snap = device_allocator[i]->snapshot();
        result.insert(result.end(), snap.begin(), snap.end());
    }
    return result;
}

c10::DeleterFnPtr PluggableAllocator::raw_deleter() const {
    return &local_raw_delete;
}

void PluggableAllocator::cacheInfo(int dev_id, size_t *cachedAndFree, size_t *largestBlock) {
    TORCH_INTERNAL_ASSERT(
        0 <= dev_id && static_cast<size_t>(dev_id) < device_allocator.size(),
        "device index out of range.");
    device_allocator[dev_id]->cache_info(cachedAndFree, largestBlock);
}

void PluggableAllocator::assertValidDevice(int device) {
    int device_num = c10_npu::device_count();
    AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats PluggableAllocator::getDeviceStats(int device) {
    assertValidDevice(device);
    return device_allocator[device]->get_stats();
}

void PluggableAllocator::resetAccumulatedStats(int device) {
    assertValidDevice(device);
    device_allocator[device]->reset_accumulated_stats();
}

void PluggableAllocator::resetPeakStats(int device) {
    assertValidDevice(device);
    device_allocator[device]->reset_peak_stats();
}

void PluggableAllocator::raw_delete(void *ptr) {
    this->free(ptr);
}

void PluggableAllocator::FreeDeviceCachedMemory(int device) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "device index out of range.");
    device_allocator[device]->empty_cache(true);
}

std::string PluggableAllocator::name() {
    return "native";
}