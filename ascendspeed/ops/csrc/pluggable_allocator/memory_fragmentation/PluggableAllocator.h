#ifndef NPU_CACHE_ALLOCATOR_PLUGGABLEALLOCATOR_H
#define NPU_CACHE_ALLOCATOR_PLUGGABLEALLOCATOR_H

#include "CachingAllocatorConfig.h"
#include "EventPool.h"
#include "DeviceCachingAllocator.h"
#include "Recorder.h"

class PluggableAllocator {
  private:
    std::mutex mutex;

    // allocated blocks by device pointer
    ska::flat_hash_map<void *, Block *> allocated_blocks;

    mutable std::mutex npu_free_mutex;
    void add_allocated_block(Block *block);

    PluggableAllocator() {}
  public:
    PluggableAllocator(const PluggableAllocator &) = delete;
    PluggableAllocator& operator=(const PluggableAllocator&) = delete;

    static PluggableAllocator& getInstance() {
        static PluggableAllocator instance;
        return instance;
    }

    std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

    std::mutex *getFreeMutex() const;
    Block *get_allocated_block(void *ptr, bool remove = false);
    void init(int device_count);
    bool initialized();
    void *malloc(int device, size_t size, aclrtStream stream);
    void free(void *ptr);
    void setMemoryFraction(double fraction, int device);
    void emptyCache(bool check_error);
    void setShutdownStats();
    void *getBaseAllocation(void *ptr, size_t *outSize);
    void recordStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream);
    void eraseStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream);
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

#endif