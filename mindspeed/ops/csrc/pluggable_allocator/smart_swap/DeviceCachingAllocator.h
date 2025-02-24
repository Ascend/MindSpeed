// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright 2022 The GLake Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "common.h"
#include "EventPool.h"
#include "CachingAllocatorConfig.h"

class DeviceCachingAllocator {
private:
    // lock around all operations
    mutable std::recursive_mutex mutex;

    // device statistics
    DeviceStats stats;

    // unallocated cached blocks larger than 1 MB
    BlockPool large_blocks;

    // unallocated cached blocks larger than 64 MB
    // BlockPool huge_blocks;

    // fused blocks that has been mapped to fragment blocks in size order
    BlockPool free_fused_blocks;

    // fused blocks that has been mapped to fragment blocks in release order
    std::unordered_map<aclrtStream, BlockEventOrderPool> free_fused_blocks_in_release_order;

    // fused blocks which is free, but it's phy_chunks are used by other block of my stream
    std::unordered_map<aclrtStream, BlockEventOrderPool> fragmented_free_fused_blocks;

    // unallocated cached blocks 1 MB or smaller
    BlockPool small_blocks;

    // allocated or in use by a stream. Holds all active allocations,
    // whether they came from graph_pools or one of the BlockPools above.
    ska::flat_hash_set<Block *> active_blocks;

    // active fused blocks
    ska::flat_hash_set<Block *> active_fused_blocks;

    // active fused blocks to be garbage collected
    ska::flat_hash_set<Block *> active_fused_blocks_to_gc;

    // captures_underway tracks if a capture might be underway on any stream.
    // Most of the time it's zero, in which case malloc can avoid calling
    // cudaStreamGetCaptureInfo in the hot path.
    int captures_underway = 0;
    // See free() for this thing's purpose
    std::vector<Block *> needs_events_deferred_until_no_capture;
    // outstanding cuda events
    ska::flat_hash_map<c10_npu::NPUStream, std::deque<std::pair<EventPool::Event, Block *>>> npu_events;

    // record used memory.
    size_t total_allocated_memory = 0;

    size_t total_fuse_size = 0;

    size_t allowed_memory_maximum = 0;

    bool set_fraction = false;

    std::atomic<CreateContextFn> context_recorder_;
    size_t alloc_trace_next = 0;
    bool alloc_trace_record_context_ = false;
    RecordContext record_context_ = RecordContext::NEVER;
    size_t alloc_trace_max_entries_ = 1;
    std::vector<TraceEntry> *alloc_trace; // pointer because we need to intentionally leak this on
                                          // deallocation it can hold references to Python state which
                                          // will already be destroyed when we are in exit handlers

    // XXX - maybe we should generalize and have multiple events
    std::vector<OutOfMemoryObserver> oom_observers_;

public:
    DeviceCachingAllocator()
        : large_blocks(BlockComparator, false),
          free_fused_blocks(BlockComparator, false),
          small_blocks(BlockComparator, true),
          alloc_trace(new std::vector<TraceEntry>())
    {
        stats.max_split_size = CachingAllocatorConfig::max_split_size();
        context_recorder_.store(nullptr);
    }

    // All public methods (except the above) acquire the allocator mutex.
    // Thus, do not call a public method from another public method.

    Block *malloc(int device, size_t orig_size, aclrtStream stream);

    Block *alloc_found_block(AllocParams params, size_t orig_size, bool split_remainder);

    void free(Block *block);

    void update_block(Block *block);

    void *getBaseAllocation(Block *block, size_t *outSize);

    void recordStream(Block *block, c10_npu::NPUStream stream);

    void eraseStream(Block *block, c10_npu::NPUStream stream);

    /* * set memory fraction to limit maximum allocated memory * */
    void setMemoryFraction(double fraction);

    /* * returns cached blocks to the system allocator * */
    void emptyCache(bool check_error);

    /* * Retrieves info (total size + largest block) of the memory cache * */
    void cacheInfo(size_t *total, size_t *largest);

    /* * Returns a copy of the memory allocator stats * */
    DeviceStats getStats();

    /* * Resets the historical accumulation stats for the device * */
    void resetAccumulatedStats();

    /* * Resets the historical peak stats for the device * */
    void resetPeakStats();

    /* * Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. * */
    std::vector<SegmentInfo> snapshot();

    static size_t round_size(size_t size);

private:
    // All private methods do not acquire the allocator mutex.

    std::vector<const Block *> get_all_blocks() const;

    /* * moves a block into a pool of cached free blocks * */
    void free_block(Block *block, bool flag);

    bool need_merge(Block *dst, Block *src);

    /* * combine previously split blocks. returns the size of the subsumed block, or 0 on failure. * */
    size_t try_merge_blocks(Block *dst, Block *src, BlockPool &pool);

    BlockPool &get_pool(size_t size);

    StatType get_stat_type_for_pool(const BlockPool &pool);

    StatTypes get_stat_types_for_pool(const BlockPool &pool);

    bool should_split(const Block *block, size_t size);

    static size_t get_allocation_size(size_t size);

    bool get_free_block(AllocParams &p);

    bool trigger_free_memory_callbacks(AllocParams &p);

    void garbage_collect_cached_blocks();

    bool realloc_block(AllocParams &p, bool isRetry);

    /* * Free one or more oversize blocks to the system allocator.  But only enough to satisfy the target size * */
    bool release_available_cached_blocks(const AllocParams &p);

    bool release_cached_blocks();

    void release_block(Block *block);

    void release_blocks(BlockPool &pool);

    EventPool::Event create_event_internal(int idx);

    void synchronize_and_free_events();

    void insert_events(Block *block);

    void insert_free_event_into_alloc_stream(Block *block);

    void insert_events_deferred_until_no_capture();

    void process_events();

    // Accumulates sizes of all memory blocks for given device in given pool
    void cache_info_aux(BlockPool &blocks, size_t *total, size_t *largest);

    bool get_fused_fragmented_blocks(AllocParams &p, int time);

    bool release_swapout_blocks();

    Block *stitch_block(std::vector<Block *> &blocks2fuse, AllocParams &p);

    Block *split_large_block(Block *block, size_t request_size);

    void release_large_block(Block *block);

    void activate_large_block(Block *block);

    void deactivate_large_block(Block *block);

    size_t garbage_collect_fused_blocks(int time, size_t require_size = 0);
};
