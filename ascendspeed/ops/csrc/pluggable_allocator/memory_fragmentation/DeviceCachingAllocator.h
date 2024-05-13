#ifndef PLUGGABLEALLOCATOR_DEVICECACHINGALLOCATOR_H
#define PLUGGABLEALLOCATOR_DEVICECACHINGALLOCATOR_H

#include "common.h"
#include "EventPool.h"
#include "CachingAllocatorConfig.h"
#include "Recorder.h"

class DeviceCachingAllocator {
private:
    // lock around all operations
    mutable std::recursive_mutex mutex;

    // device statistics
    DeviceStats stats;

    struct LcPool {
        BlockPool large_blocks;
        BlockPool small_blocks;
        LcPool(BlockPoolType type) : large_blocks(false, type), small_blocks(true, type) {}
    } long_lc_pools, default_lc_pools;

    MemoryRecorder recorder;

    MallocRecorder malloc_recorder;

    // allocated or in use by a stream
    ska::flat_hash_set<Block*> active_blocks;

    // outstanding acl events
    ska::flat_hash_map<c10_npu::NPUStream, std::deque<std::pair<EventPool::Event, Block*>>> npu_events;

    // record used memory.
    size_t total_allocated_memory = 0;

    // record maximum allowed memory.
    size_t allowed_memory_maximum = 0;

    // all live expandable segments
    std::vector<ExpandableSegment*> expandable_segments_;

    bool set_fraction = false;

    // whether shutdown.
    bool shutdown_stats = false;

public:
    // The last three blocks receive the tensor to prevent memory conflicts.
    static constexpr int prevent_memory_conflict_num = 3;
    // A pre-judgment is made for the memory decrease caused by too many malloc failures.
    static constexpr int memory_fail_prejudgment = 209715200;

    DeviceCachingAllocator();

    void print_memory_analysis();

    // All public methods (except the above) acquire the allocator mutex.
    // Thus, do not call a public method from another public method.

    Block* malloc(int device, size_t orig_size, aclrtStream stream);

    Block* malloc_internal(int device, size_t orig_size, aclrtStream stream, LifeCycleType lc, size_t tensor_forward_end,
                           size_t tensor_forward_start);

    Block* alloc_found_block(AllocParams params, size_t orig_size, bool split_remainder);

    void free(Block* block);

    void* get_base_allocation(Block* block, size_t* outSize);

    void record_stream(Block* block, c10_npu::NPUStream stream);

    void erase_stream(Block* block, c10_npu::NPUStream stream);

    /** set memory fraction to limit maximum allocated memory **/
    void set_memory_fraction(double fraction);

    /** returns cached blocks to the system allocator **/
    void empty_cache(bool check_error);

    void dev_set_shutdown_stats();

    /** Retrieves info (total size + largest block) of the memory cache **/
    void cache_info(size_t* total, size_t* largest);

    /** Returns a copy of the memory allocator stats **/
    DeviceStats get_stats();

    /** Resets the historical accumulation stats for the device **/
    void reset_accumulated_stats();

    /** Resets the historical peak stats for the device **/
    void reset_peak_stats();

    /** Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. **/
    std::vector<SegmentInfo> snapshot() const;

    static size_t round_size(size_t size);

private:
    // All private methods do not acquire the allocator mutex.

    std::vector<const Block*> get_all_blocks() const;

    // returns the smallest possible address in any segment
    // where there is enough free address space to fit size
    // may be composed of free and unmapped segments
    Block* find_expandable_block(int device, aclrtStream stream, BlockPool* pool, size_t size);

    bool map_block(Block* to_map, size_t size);

    Block* try_allocate_expandable_block(int device, aclrtStream stream, BlockPool* pool, size_t size);

    /** moves a block into a pool of cached free blocks **/
    void free_block(Block* block);

    /** combine previously split blocks. returns the size of the subsumed block, or 0 on failure. **/
    size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool);

    LcPool& get_lc_pool(LifeCycleType lc);

    BlockPool& get_pool(size_t size, LifeCycleType lc);

    std::vector<BlockPool *> get_pool_list(size_t size, LifeCycleType lc);

    StatTypes get_stat_types_for_pool(const BlockPool& pool);

    bool should_split(const Block* block, size_t size);

    static size_t get_allocation_size(size_t size, LifeCycleType lc = LifeCycleType::DEFAULT_LC);

    bool get_free_block(AllocParams& p);

    bool get_free_block_memory_optimize(AllocParams &p, size_t tensor_forward_end, size_t tensor_step_end,
                                        size_t tensor_forward_start, size_t tensor_step_start);

    bool get_free_block_after_alloc(AllocParams& p);

    bool trigger_free_memory_callbacks(AllocParams& p);

    void garbage_collect_cached_blocks();

    bool alloc_block(AllocParams& p, bool isRetry);

    /** Free one or more oversize blocks to the system allocator.  But only enough to satisfy the target size **/
    bool release_available_cached_blocks(const AllocParams& p);

    bool release_cached_blocks(bool check_error);

    void release_expandable_segment(Block* block);

    bool release_cached_blocks_long(bool check_error);

    bool release_cached_blocks_default(bool check_error);

    void release_block(Block* block);

    void unmap_block(Block* block);

    void release_blocks(BlockPool& pool);

    EventPool::Event create_event_internal(int idx);

    void synchronize_and_free_events(bool check_error);

    void insert_events(Block* block);

    void process_events();

    // Accumulates sizes of all memory blocks for given device in given pool
    void cache_info_aux(BlockPool& blocks, size_t* total, size_t* largest);
};

#endif
