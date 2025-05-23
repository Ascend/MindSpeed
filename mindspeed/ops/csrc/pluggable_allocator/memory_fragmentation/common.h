#ifndef NPU_CACHE_ALLOCATOR_COMMON_H
#define NPU_CACHE_ALLOCATOR_COMMON_H

#include <mutex>
#include <unordered_map>
#include <climits>
#include <vector>

#include <c10/util/flat_hash_map.h>
#include "acl_base.h"
#include "acl_rt.h"

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <torch_npu/csrc/core/npu/NPUException.h>

#define MEMORY_RECORDER_DEBUG

static std::mutex alr_lock;
static std::unordered_map<void *, size_t> alr_recorder;
static size_t alr_total_size = 0;
static size_t alr_get_max() {
    static size_t rc = 0;
    if (rc == 0) {
        char *e = getenv("ALR_MAX");
        if (e) {
            std::string s(e);
            rc = std::stoll(s);
        } else {
            rc = LLONG_MAX;
        }
        printf("ALR MAX SET  %lu\n", rc);
    }
    return rc;
}

static aclError aclrtMalloc_wrapper(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
    alr_lock.lock();
    if (alr_total_size + size > alr_get_max()) {
        alr_lock.unlock();
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    alr_lock.unlock();

    auto rc = aclrtMallocAlign32(devPtr, size, policy);
    if (rc != ACL_ERROR_NONE) {
        return rc;
    }
    alr_lock.lock();
    alr_recorder[*devPtr] = size;
    alr_total_size += size;
    alr_lock.unlock();
    return rc;
}

static aclError aclrtFree_wrapper(void *devPtr) {
    alr_lock.lock();
    alr_total_size -= alr_recorder[devPtr];
    alr_recorder.erase(devPtr);
    alr_lock.unlock();
    return aclrtFree(devPtr);
}

struct Stat {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

enum struct StatType : uint64_t {
    AGGREGATE = 0,
    SMALL_POOL = 1,
    LARGE_POOL = 2,
    NUM_TYPES = 3  // remember to update this whenever a new stat type is added
};

// Struct containing info of an allocation block (i.e. a fractional part of a cudaMalloc)..
struct BlockInfo {
    int64_t size = 0;
    int64_t requested_size = 0;
    int32_t gc_counter = 0;
    bool allocated = false;
    bool active = false;
};

enum BlockPoolType : int {
    BLOCK_POOL_DEFAULT,
    BLOCK_POOL_SHORT,
    BLOCK_POOL_LONG,
};

struct SegmentInfo {
    int64_t device = 0;
    uintptr_t  address = 0;
    int64_t total_size = 0;
    int64_t requested_size = 0;
    int64_t allocated_size = 0;
    int64_t active_size = 0;
    bool is_large = false;
    bool is_expandable = false;
#ifdef MEMORY_RECORDER_DEBUG
    BlockPoolType type;
    std::string type_str;
#endif
    std::vector<BlockInfo> blocks;
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

struct DeviceStats {
    // COUNT: allocations requested by client code
    StatArray allocation;
    // COUNT: number of allocated segments from npuMalloc().
    StatArray segment;
    // COUNT: number of active memory blocks (allocated or used by stream)
    StatArray active;
    // COUNT: number of inactive, split memory blocks (unallocated but can't be released via npuFree)
    StatArray inactive_split;

    // SUM: bytes requested by client code
    StatArray allocated_bytes;
    // SUM: bytes reserved by this memory allocator (both free and used)
    StatArray reserved_bytes;
    // SUM: bytes within active memory blocks
    StatArray active_bytes;
    // SUM: bytes within inactive, split memory blocks
    StatArray inactive_split_bytes;
    // SUM: bytes requested by client code
    StatArray requested_bytes;

    // COUNT: total number of failed calls to NPU malloc necessitating cache flushes.
    int64_t num_alloc_retries = 0;

    // COUNT: total number of OOMs (i.e. failed calls to NPU after cache flush)
    int64_t num_ooms = 0;

    // COUNT: total number of oversize blocks allocated from pool
    Stat oversize_allocations;

    // COUNT: total number of oversize blocks requiring malloc
    Stat oversize_segments;

    // SIZE: maximum block size that is allowed to be split.
    int64_t max_split_size = 0;
};

using stream_set = ska::flat_hash_set<c10_npu::NPUStream>;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

static void update_stat(Stat &stat, int64_t amount) {
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }
    if (amount < 0) {
        stat.freed += -amount;
    }
}

static void reset_accumulated_stat(Stat& stat) {
    stat.allocated = 0;
    stat.freed = 0;
}

static void reset_peak_stat(Stat& stat) { stat.peak = stat.current; }

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
    for (const auto stat_type : c10::irange(stat_types.size())) {
        if (stat_types[stat_type]) {
            f(stat_type);
        }
    }
}

static void update_stat_array(StatArray& stat_array, int64_t amount, const StatTypes& stat_types) {
    for_each_selected_stat_type(stat_types,
                                [&stat_array, amount](size_t stat_type) { update_stat(stat_array[stat_type], amount); });
}

struct Block;
using Comparison = bool (*)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool{
    std::set<Block*, Comparison> blocks;
    std::set<Block*, Comparison> unmapped;
    const bool is_small;
    const BlockPoolType type;

    BlockPool(bool small, BlockPoolType type)
            : blocks(BlockComparatorSize), unmapped(BlockComparatorAddress), is_small(small), type(type) {}
};

static std::string get_block_pool_str(BlockPoolType type) {
    switch (type) {
        case BLOCK_POOL_DEFAULT:
            return "default";
        case BLOCK_POOL_LONG:
            return "long";
        case BLOCK_POOL_SHORT:
            return "short";
        default:
            return "unknown";
    }
    AT_ASSERT(0);
    return "";
}

struct ExpandableSegment;

struct Block {
    int device; // npu
    aclrtStream stream; // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size; // block size in bytes
    size_t requested_size; // memory originally requested
    BlockPool* pool; // owning memory pool
    void* ptr; // memory address
    bool allocated; // in-use flag
    bool mapped{true}; // is the virtual address range this Block references
    // backed by physical pages. Always true when
    // expandable_segment_ is null. When false
    // This Block will be aligned to the segment size
    // of its expandable_segment_.
    Block* prev; // prev block if split from a larger allocation
    Block* next; // next block if split from a larger allocation
    int event_count; // number of outstanding NPU events
    size_t start_tik{0}; // Record the time when the block was created during the step phase
    size_t forward_start_tik{0}; // Record the time when the block was created in the forward phase
    size_t tensor_size{0}; // Tensor_size is the size processed by orig_size
    size_t orig_size{0}; // origin tensor size
    int step_count{0}; // how many steps have passed （Record how many steps the current block has passed）
    int forward_count{0}; // Record how many forwards have been passed
    bool in_step{0}; // Determine whether the current block is in the step stage
    bool in_forward{0}; // Determine if the current block is in the forward stage
    int gc_count{0}; // counter for prioritizing older / less useful blocks for
    // garbage collection
    ExpandableSegment* expandable_segment_{nullptr};

    Block(int device, aclrtStream stream, size_t size, BlockPool* pool, void* ptr)
            : device(device),
              stream(stream),
              stream_uses(),
              size(size),
              requested_size(0),
              pool(pool),
              ptr(ptr),
              allocated(0),
              prev(nullptr),
              next(nullptr),
              event_count(0),
              gc_count(0) {}

    // constructor for search key
    Block(int device, aclrtStream stream, size_t size)
            : device(device),
              stream(stream),
              stream_uses(),
              size(size),
              requested_size(0),
              pool(nullptr),
              ptr(nullptr),
              allocated(0),
              prev(nullptr),
              next(nullptr),
              event_count(0),
              gc_count(0) {}

    bool is_split() const { return (prev != nullptr) || (next != nullptr); }

    void splice(Block* before, Block* after) {
        if (before) {
            //  TORCH_INTERNAL_ASSERT(before->next == after);
            before->next = this;
        }
        prev = before;
        if (after) {
            //  TORCH_INTERNAL_ASSERT(after->prev == before);
            after->prev = this;
        }
        next = after;
    }
};

struct SegmentRange {
    char* ptr;
    size_t size;
    SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

struct ExpandableSegment {
    ExpandableSegment(int device, aclrtStream stream, size_t size)
            : device_(device),
              stream_(stream),
              max_handles_(0),
            // 2MB for small pool, 20MB for large pool
              segment_size_(size) {
        size_t device_free;
        size_t device_total;
        aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total);
        // we allocate enough address space for 1 1/8 the total memory on the NPU.
        // This allows for some cases where we have to unmap pages earlier in the
        // segment to put them at the end.
        max_handles_ = numSegments(device_total + device_total / 8);
        aclrtReserveMemAddress(&ptr_, segment_size_ * max_handles_, 0, NULL, 1);
    }
    // begin must be aligned to segment_size_.
    // returns the actual range mapped, which may be
    // greater than requested if size is not aligned to segment_size_.
    // return size of 0 indicates OOM
    SegmentRange map(SegmentRange range) {
        auto begin = segmentLeft(range.ptr);
        auto end = segmentRight(range.ptr + range.size);
        //  TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
        if (begin == end) {
            return rangeFromHandles(begin, end);
        }
        while (end > handles_.size()) {
            handles_.emplace_back(c10::nullopt);
        }
        for (auto i : c10::irange(begin, end)) {
            //  TORCH_INTERNAL_ASSERT(!handles_.at(i));
            aclrtDrvMemHandle handle = nullptr;
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr = ACL_HBM_MEM_HUGE;
            prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device_;
            prop.reserve = 0;
            auto status = aclrtMallocPhysical(&handle, segment_size_, &prop, 0);
            if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
                for (auto j : c10::irange(begin, i)) {
                    auto h = handles_.at(j).value();
                    handles_.at(j) = c10::nullopt;
                    aclrtFreePhysical(h);
                }
                trimHandles();
                return rangeFromHandles(begin, begin);
            }
            handles_.at(i) = handle;
        }
        for (auto i : c10::irange(begin, end)) {
            aclrtMapMem(ptr_ + i * segment_size_, segment_size_, 0, handles_.at(i).value(), 0);
        }
        return rangeFromHandles(begin, end);
    }

    // unmaps all the completely empty segment_size_ segments between
    // [begin, begin + size), returns the offset where the range begin,
    // and the actual size unmapped (multiple of segment_size_)
    SegmentRange unmap(SegmentRange range) {
        auto begin = segmentRight(range.ptr);
        auto end = segmentLeft(range.ptr + range.size);
        if (begin >= end) {
            return SegmentRange{range.ptr, 0};
        }
        unmapHandles(begin, end);
        return rangeFromHandles(begin, end);
    }

    char* ptr() const { return (char*)ptr_; }

    size_t size() const { return max_handles_ * segment_size_; }

    ~ExpandableSegment() {
        forEachAllocatedRange([&](size_t begin, size_t end) { unmapHandles(begin, end); });
        aclrtReleaseMemAddress(ptr_);
    }

private:
    void unmapHandles(size_t begin, size_t end) {
        // note: unlike aclrtFree, MemUnmap and MemRelease do
        // not appear to synchronize in all cases, so we have to wait for the
        // stream to finish before this memory is truly free.

        // cannot call c10::npu::stream_synchronize because
        // it might grab the GIL which can lead to a deadlock
        // Locking order must be GIL -> Allocator Lock
        aclrtSynchronizeStream(stream_);
        for (auto i : c10::irange(begin, end)) {
            aclrtDrvMemHandle h = handles_.at(i).value();
            handles_.at(i) = c10::nullopt;
            aclrtUnmapMem(ptr_ + segment_size_ * i);
            aclrtFreePhysical(h);
        }
        trimHandles();
    }

    void trimHandles() {
        while (!handles_.empty() && !handles_.back()) {
            handles_.pop_back();
        }
    }

    void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
        auto start = 0;
        for (auto i : c10::irange(handles_.size())) {
            if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
                start = i;
            }
            if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
                fn(start, i + 1);
            }
        }
    }

    size_t numSegments(size_t size) { return (size + segment_size_ - 1) / segment_size_; }

    size_t segmentLeft(char* p) {
        auto size = p - ptr();
        return size / segment_size_;
    }

    size_t segmentRight(char* p) {
        auto size = p - ptr();
        return numSegments(size);
    }

    SegmentRange rangeFromHandles(size_t begin, size_t end) {
        return SegmentRange(ptr() + segment_size_ * begin, segment_size_ * (end - begin));
    }

    int device_;
    aclrtStream stream_;
    void* ptr_{};
    size_t max_handles_;
    size_t segment_size_;
    std::vector<c10::optional<aclrtDrvMemHandle>> handles_;
};

static bool BlockComparatorSize(const Block* a, const Block* b) {
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

static bool BlockComparatorAddress(const Block* a, const Block* b) {
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

static std::string format_size(uint64_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KiB";
    } else if (size <= 1073741824ULL) {
        os << (size / 1048576.0);
        os << " MiB";
    } else {
        os << (size / 1073741824.0);
        os << " GiB";
    }
    return os.str();
}

struct AllocParams {
    AllocParams(int device, size_t size, aclrtStream stream, BlockPool* pool, size_t alloc_size, DeviceStats& stats)
            : search_key(device, stream, size), pool(pool), alloc_size(alloc_size), block(nullptr), err(ACL_ERROR_NONE) {}

    AllocParams() = default;

    int device() const { return search_key.device; }
    aclrtStream stream() const { return search_key.stream; }
    size_t size() const { return search_key.size; }

    Block search_key;
    BlockPool* pool;
    size_t alloc_size;
    Block* block;
    StatTypes stat_types = {false};
    aclError err;
};

#endif
