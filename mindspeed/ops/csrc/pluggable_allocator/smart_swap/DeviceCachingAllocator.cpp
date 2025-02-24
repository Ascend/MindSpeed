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
#include "DeviceCachingAllocator.h"

#include <chrono>

#include "swap_log.h"
#include "NPUSwapManager.h"

Block *DeviceCachingAllocator::malloc(int device, size_t orig_size, aclrtStream stream)
{
    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (device == -1) {
        SWAP_CHECK_ERROR(c10_npu::GetDevice(&device));
    }
    // process outstanding npuEvents
    process_events();
    auto size = round_size(orig_size);
    auto &pool = get_pool(size);

    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

    AllocParams swap_params(device, size, stream, &pool, alloc_size, stats);
    swap_params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params) ||
        // Trigger callbacks and retry search
        (trigger_free_memory_callbacks(params) && get_free_block(params)) ||
        get_fused_fragmented_blocks(params, 0);
    if (!block_found) {
        // Do garbage collection if the flag is set.
        if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
            garbage_collect_cached_blocks();
        }
        if (c10_npu::swap::NPUSwapManager::GetInstance().swap_enable) {
            block_found = realloc_block(params, false);
            if (!block_found) {
                block_found = (release_swapout_blocks() &&
                    (get_free_block(swap_params) || get_fused_fragmented_blocks(swap_params, 1)));
                if (block_found) {
                    params.err = swap_params.err;
                    params.block = swap_params.block;
                } else {
                    block_found = realloc_block(params, true) ||
                        (release_available_cached_blocks(params) && realloc_block(params, true)) ||
                        // Free all non-split cached blocks and retry alloc.
                        (release_cached_blocks() && realloc_block(params, true)) ||
                        get_fused_fragmented_blocks(params, 2);
                }
            }
        } else {
            // Attempt allocate
            block_found = realloc_block(params, false) ||
                // Free enough available cached blocks to satisfy alloc and retry alloc.
                ((release_swapout_blocks() || release_available_cached_blocks(params)) &&
                 realloc_block(params, true)) ||
                get_fused_fragmented_blocks(params, 1) ||
                // Free all non-split cached blocks and retry alloc.
                (C10_LIKELY(captures_underway == 0) && release_cached_blocks() && realloc_block(params, true)) ||
                get_fused_fragmented_blocks(params, 2);
        }
    }

    if (!block_found) {
        if (params.err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
            if (c10_npu::swap::NPUSwapManager::GetInstance().swap_oom_enable) {
                SWAP_LOG_WARN("[SwapOomEnable] Trigger OOM error when malloc()");
                c10_npu::swap::NPUSwapManager::GetInstance().config.isOOM = true;
                if (!active_blocks.empty() &&
                    !c10_npu::swap::NPUSwapManager::GetInstance().GetStorageImplMap().empty()) {
                    Block *findBlock = nullptr;
                    for (std::deque<void *>::iterator itQ =
                             c10_npu::swap::NPUSwapManager::GetInstance().GetTensorQueue().begin();
                        itQ != c10_npu::swap::NPUSwapManager::GetInstance().GetTensorQueue().end();) {
                        auto it = std::find_if(active_fused_blocks.begin(), active_fused_blocks.end(),
                            [&itQ](const Block *block) { return block->ptr == *itQ; });
                        if (it != active_fused_blocks.end()) {
                            if (!c10_npu::swap::NPUSwapManager::GetInstance().config.enableCustomRecordStream) {
                                c10_npu::npuSynchronizeDevice(true);
                                eraseStream((*it), c10_npu::swap::NPUSwapManager::GetInstance().GetSwapStream());
                            }
                            if ((*it)->stream_uses.empty()) {
                                // 该判断实现的效果是找到比alloc_size大的最小的block，如果没有block比alloc_size大，则会找到比alloc_size小的最大的block
                                if (findBlock == nullptr || // 1.如果当前为遍历到的第一个block，直接赋给findBlock
                                    (findBlock->size < alloc_size && (*it)->size >
                                    findBlock->size) || // 2.如果已经找到的block大小比alloc_size小，只要遍历到的block比其大小大，就更新findBlock
                                    (findBlock->size >= alloc_size && (*it)->size >= alloc_size &&
                                    (*it)->size <
                                    findBlock->size)) { // 3.如果已经找到的block大小比alloc_size大，那么只有在当前遍历到的block大小不小于alloc_size，又比已经找到的block小，才更新findBlock
                                    findBlock = *it;
                                }
                            }
                            ++itQ;
                        } else {
                            auto it = std::find_if(active_blocks.begin(), active_blocks.end(),
                                [&itQ](const Block *block) { return block->ptr == *itQ; });
                            if (it != active_blocks.end()) {
                                if (!c10_npu::swap::NPUSwapManager::GetInstance().config.enableCustomRecordStream) {
                                    c10_npu::npuSynchronizeDevice(true);
                                    eraseStream((*it), c10_npu::swap::NPUSwapManager::GetInstance().GetSwapStream());
                                }
                                if ((*it)->stream_uses.empty()) {
                                    // 该判断实现的效果是找到比alloc_size大的最小的block，如果没有block比alloc_size大，则会找到比alloc_size小的最大的block
                                    if (findBlock == nullptr || // 1.如果当前为遍历到的第一个block，直接赋给findBlock
                                        (findBlock->size < alloc_size && (*it)->size >
                                        findBlock->size) || // 2.如果已经找到的block大小比alloc_size小，只要遍历到的block比其大小大，就更新findBlock
                                        (findBlock->size >= alloc_size && (*it)->size >= alloc_size &&
                                        (*it)->size <
                                        findBlock->size)) { // 3.如果已经找到的block大小比alloc_size大，那么只有在当前遍历到的block大小不小于alloc_size，又比已经找到的block小，才更新findBlock
                                        findBlock = *it;
                                    }
                                }
                                ++itQ;
                            } else {
                                c10_npu::swap::NPUSwapManager::GetInstance().GetStorageImplMap().erase(*itQ);
                                itQ = c10_npu::swap::NPUSwapManager::GetInstance().GetTensorQueue().erase(itQ);
                            }
                        }
                    }

                    if (findBlock != nullptr) {
                        SWAP_LOG_WARN("[SwapOomEnable] malloc OOM, need swap out ptrInBlock, size[%zu]",
                            findBlock->size);
                        throw c10_npu::swap::SwapOutOfMemError("malloc OOM, need swap out ptrInBlock.", findBlock->ptr);
                    }
                }
            }
            // For any error code other than ACL_ERROR_RT_MEMORY_ALLOCATION,
            // alloc_block should have thrown an exception already.
            TORCH_INTERNAL_ASSERT(params.err == ACL_ERROR_RT_MEMORY_ALLOCATION);

            size_t device_free;
            size_t device_total;
            SWAP_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
            std::string allowed_info;

            if (set_fraction) {
                allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
            }

            stats.num_ooms += 1;
            auto observers_local = oom_observers_;
            lock.unlock();

            for (const auto &obs : observers_local) {
                obs(device, alloc_size, set_fraction ? allowed_memory_maximum : device_total, device_free);
            }
            // "total capacity": total global memory on GPU
            // "allowed": memory is allowed to use, which set by fraction.
            // "already allocated": memory allocated by the program using the
            //                      caching allocator
            // "free": free memory as reported by the CUDA API
            // "cached": memory held by the allocator but not used by the program
            //
            // The "allocated" amount  does not include memory allocated outside
            // of the caching allocator, such as memory allocated by other programs
            // or memory held by the driver.
            //
            // The sum of "allocated" + "free" + "cached" may be less than the
            // total capacity due to memory held by the driver and usage by other
            // programs.
            //
            // Note that at this point free_cached_blocks has already returned all
            // possible "cached" memory to the driver. The only remaining "cached"
            // memory is split from a larger block that is partially in-use.
            AT_ERROR("NPU out of memory. Tried to allocate ", format_size(alloc_size), " (NPU ", device, "; ",
                format_size(device_total), " total capacity; ",
                format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                " already allocated; ",
                format_size(stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current), " current active; ",
                format_size(device_free), " free; ", allowed_info,
                format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                " reserved in total by PyTorch)",
                " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.");
        } else {
            SWAP_CHECK_ERROR(params.err);
        }
    }

    Block *block = params.block;
    Block *remaining = nullptr;

    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    const bool already_split = block->is_split();

    if (pool.is_small && should_split(block, size)) {
        remaining = block;

        block = new Block(device, stream, size, &pool, block->ptr);
        block->prev = remaining->prev;
        if (block->prev) {
            block->prev->next = block;
        }
        block->next = remaining;

        remaining->prev = block;
        remaining->ptr = static_cast<char *>(remaining->ptr) + size;
        remaining->size -= size;

        bool inserted = pool.blocks.insert(remaining).second;

        if (already_split) {
            // An already-split inactive block is being shrunk by size bytes.
            update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
        } else {
            // A new split inactive block is being created from a previously unsplit
            // block, size remaining->size bytes.
            for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                update_stat(stats.inactive_split_bytes[stat_type], remaining->size);
                update_stat(stats.inactive_split[stat_type], 1);
            });
        }
    } else if (already_split) {
        // An already-split block is becoming active
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
            update_stat(stats.inactive_split_bytes[stat_type], -static_cast<std::int64_t>(block->size));
            update_stat(stats.inactive_split[stat_type], -1);
        });
    }

    block->allocated = true;
    block->requested_size = orig_size;
    block->actual_size = size;

    bool inserted = false;
    if (block->vmm_segment && block->vmm_segment->fused) {
        active_fused_blocks.insert(block);
    } else {
        inserted = active_blocks.insert(block).second;
    }

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.allocation[stat_type], 1);
        update_stat(stats.allocated_bytes[stat_type], static_cast<std::int64_t>(block->actual_size));
        update_stat(stats.requested_bytes[stat_type], static_cast<std::int64_t>(block->requested_size));
        update_stat(stats.active[stat_type], 1);
        update_stat(stats.active_bytes[stat_type], block->size);
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_allocations, 1);

    c10_npu::swap::NPUSwapManager::GetInstance().tensorPtrCountMap[reinterpret_cast<size_t>(block->ptr)]++;

    return block;
}


void DeviceCachingAllocator::free(Block *block)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = { false };
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.allocation[stat_type], -1);
        update_stat(stats.allocated_bytes[stat_type], -static_cast<std::int64_t>(block->actual_size));
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
        if (C10_UNLIKELY(captures_underway)) {
            needs_events_deferred_until_no_capture.push_back(block);
        } else {
            insert_events(block);
        }
    } else {
        insert_free_event_into_alloc_stream(block);
        update_block(block);
    }
}

void DeviceCachingAllocator::update_block(Block *block)
{
    block->allocated = false;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    bool flag = false;

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto &pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;
    StatTypes stat_types = { false };
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.inactive_split[stat_type], net_change_inactive_split_blocks);
        update_stat(stats.inactive_split_bytes[stat_type], net_change_inactive_split_size);
        update_stat(stats.active[stat_type], -1);
        update_stat(stats.active_bytes[stat_type], -static_cast<std::int64_t>(original_block_size));
        if (!flag) {
            update_stat(stats.requested_bytes[stat_type], -static_cast<std::int64_t>(requested_size));
        }
    });

    if (block->pool->is_small) {
        free_block(block, flag);
    } else {
        deactivate_large_block(block);
    }
}

void *DeviceCachingAllocator::getBaseAllocation(Block *block, size_t *outSize)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
        block = block->prev;
    }
    void *basePtr = block->ptr;
    if (outSize) {
        size_t size = 0;
        while (block) {
            size += block->size;
            block = block->next;
        }
        *outSize = size;
    }
    return basePtr;
}

void DeviceCachingAllocator::recordStream(Block *block, c10_npu::NPUStream stream)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    block->stream_uses.insert(stream);
}

void DeviceCachingAllocator::eraseStream(Block *block, c10_npu::NPUStream stream)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    block->stream_uses.erase(stream);

    // free block, lazy destory block related events
    for (auto it = npu_events[stream].begin(); it != npu_events[stream].end();) {
        if (block != it->second) {
            it++;
            continue;
        }
        it = npu_events[stream].erase(it);
        block->event_count--;
        if (block->event_count == 0) {
            update_block(block);
            break;
        }
    }
}

void DeviceCachingAllocator::setMemoryFraction(double fraction)
{
    size_t device_free;
    size_t device_total;
    aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total);
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
}

void DeviceCachingAllocator::emptyCache(bool check_error)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
    size_t garbage_size = garbage_collect_fused_blocks(2, 0);
}

void DeviceCachingAllocator::cacheInfo(size_t *total, size_t *largest)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cache_info_aux(large_blocks, total, largest);
    cache_info_aux(small_blocks, total, largest);
}

DeviceStats DeviceCachingAllocator::getStats()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
}

void DeviceCachingAllocator::resetAccumulatedStats()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
        reset_accumulated_stat(stats.allocation[statType]);
        reset_accumulated_stat(stats.segment[statType]);
        reset_accumulated_stat(stats.active[statType]);
        reset_accumulated_stat(stats.inactive_split[statType]);
        reset_accumulated_stat(stats.allocated_bytes[statType]);
        reset_accumulated_stat(stats.reserved_bytes[statType]);
        reset_accumulated_stat(stats.active_bytes[statType]);
        reset_accumulated_stat(stats.inactive_split_bytes[statType]);
        reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
}

/* * Resets the historical peak stats for the device * */
void DeviceCachingAllocator::resetPeakStats()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
        reset_peak_stat(stats.allocation[statType]);
        reset_peak_stat(stats.segment[statType]);
        reset_peak_stat(stats.active[statType]);
        reset_peak_stat(stats.inactive_split[statType]);
        reset_peak_stat(stats.allocated_bytes[statType]);
        reset_peak_stat(stats.reserved_bytes[statType]);
        reset_peak_stat(stats.active_bytes[statType]);
        reset_peak_stat(stats.inactive_split_bytes[statType]);
        reset_peak_stat(stats.requested_bytes[statType]);
    }

    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
}

std::vector<SegmentInfo> DeviceCachingAllocator::snapshot()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);

    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();
    for (const Block * const head_block : all_blocks) {
        if (head_block->prev != nullptr) {
            continue;
        }
        result.emplace_back();
        SegmentInfo &segment_info = result.back();
        segment_info.device = head_block->device;
        segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
        segment_info.stream = head_block->stream;
        segment_info.is_large = (!head_block->pool->is_small);

        const Block *block = head_block;
        while (block != nullptr) {
            segment_info.blocks.emplace_back();
            auto &block_info = segment_info.blocks.back();

            block_info.size = block->size;
            block_info.requested_size = block->requested_size;
            block_info.allocated = block->allocated;
            block_info.active = block->allocated || (block->event_count > 0) || !block->stream_uses.empty();

            segment_info.total_size += block_info.size;
            if (block_info.allocated) {
                segment_info.allocated_size += block_info.size;
            }
            if (block_info.active) {
                segment_info.active_size += block_info.size;
                segment_info.requested_size += block_info.requested_size;
            }
            block = block->next;
        }
        total_active += segment_info.active_size;
    }

    std::sort(result.begin(), result.end(),
        [](const SegmentInfo &a, const SegmentInfo &b) { return a.address < b.address; });

    return result;
}

size_t DeviceCachingAllocator::round_size(size_t size)
{
    size = size + 32;
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        size_t block_round_size = kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        if (block_round_size > kSmallSize) {
            // if block will alloc from large_blocks, round to 2M
            block_round_size = kGranularity * ((size + kGranularity - 1) / kGranularity);
        }
        return block_round_size;
    }
}

std::vector<const Block *> DeviceCachingAllocator::get_all_blocks() const
{
    std::vector<const Block *> blocks;
    blocks.insert(blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
}

void DeviceCachingAllocator::free_block(Block *block, bool flag)
{
    TORCH_INTERNAL_ASSERT(!block->allocated && block->event_count == 0 && block->stream_uses.empty());
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto &pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block *, 2> merge_candidates = { block->prev, block->next };
    for (Block *merge_candidate : merge_candidates) {
        const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
        if (subsumed_size > 0) {
            net_change_inactive_split_blocks -= 1;
            net_change_inactive_split_size -= subsumed_size;
        }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it back into.
    bool inserted = pool.blocks.insert(block).second;

    TORCH_INTERNAL_ASSERT(inserted);
    if (vmmDefragment > 0 && block->vmm_segment /* !pool.is_small */) {
        for (size_t i = 0; i < block->vmm_segment->phy_chunks.size(); i++) {
            auto &p = block->vmm_segment->phy_chunks[i];
            p->free = true;
        }
        block->vmm_segment->num_free_chunks = block->vmm_segment->phy_chunks.size();
        block->vmm_segment->num_used_chunks = 0;
    }

    if (block->is_split()) {
        net_change_inactive_split_blocks += 1;
        net_change_inactive_split_size += block->size;
    }
}

bool DeviceCachingAllocator::need_merge(Block *dst, Block *src)
{
    if (!src || src->allocated || src->event_count > 0 || !src->stream_uses.empty()) {
        return false;
    }
    return true;
}

size_t DeviceCachingAllocator::try_merge_blocks(Block *dst, Block *src, BlockPool &pool)
{
    if (!src || src->allocated || src->event_count > 0 || !src->stream_uses.empty()) {
        return 0;
    }
    if (src->vmm_segment && src->vmm_segment->phy_chunks[0]->mapped_blocks.size() > 1) {
        return 0;
    }
    if (dst->vmm_segment && dst->vmm_segment->phy_chunks[0]->mapped_blocks.size() > 1) {
        return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
        dst->ptr = src->ptr;
        dst->prev = src->prev;
        if (dst->prev) {
            dst->prev->next = dst;
        }
        if (!dst->history) {
            dst->history = std::move(src->history);
            dst->history_last = src->history_last;
        } else if (src->history) {
            src->history_last->next = std::move(dst->history);
            dst->history = std::move(src->history);
        }
        src->history_last = nullptr;
    } else { // [dest src]
        dst->next = src->next;
        if (dst->next) {
            dst->next->prev = dst;
        }

        if (!dst->history) {
            dst->history = std::move(src->history);
            dst->history_last = src->history_last;
        } else if (src->history) {
            dst->history_last->next = std::move(src->history);
            dst->history_last = src->history_last;
        }
        src->history_last = nullptr;
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();
    if (vmmDefragment > 0 && dst->vmm_segment) {
        bool ret = dst->vmm_segment->remerge(*(src->vmm_segment));
        size_t offset = 0;
        for (auto &phy_block : dst->vmm_segment->phy_chunks) {
            phy_block->mapped_blocks[0].block = dst;
            phy_block->mapped_blocks[0].offset = offset;
            offset++;
        }
    }

    delete src;
    return subsumed_size;
}

BlockPool &DeviceCachingAllocator::get_pool(size_t size)
{
    if (size <= kSmallSize) {
        return small_blocks;
    } else {
        return large_blocks;
    }
}

bool DeviceCachingAllocator::should_split(const Block *block, size_t size)
{
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
        return remaining >= kMinBlockSize;
    } else {
        return (size < CachingAllocatorConfig::max_split_size()) && (remaining >= kGranularity);
    }
}

StatType DeviceCachingAllocator::get_stat_type_for_pool(const BlockPool &pool)
{
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
}

StatTypes DeviceCachingAllocator::get_stat_types_for_pool(const BlockPool &pool)
{
    StatTypes stat_types = { false };
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
}

size_t DeviceCachingAllocator::get_allocation_size(size_t size)
{
    if (size <= kSmallSize) {
        return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
        return kLargeBuffer;
    } else {
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

bool DeviceCachingAllocator::get_free_block(AllocParams &p)
{
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    static const double reuseLimit = ([]() -> double {
        const char *env = getenv("reuseLimit");
        if (env) {
            return atof(env);
        } else {
            return 1.0f;
        }
    })();

    static const size_t fragment_limit = ([]() -> size_t {
        const char *env = getenv("fragLimit");
        if (env) {
            return static_cast<size_t>(std::stoll(env));
        } else {
            return static_cast<size_t>(16777216);
        }
    })();

    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    BlockPool &pool = *p.pool;
    if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        // Track block reuse interval only when garbage collection is enabled.
        for (auto &b : pool.blocks) {
            ++b->gc_count;
        }
    }

    if (vmmDefragment > 0 && !pool.is_small && p.search_key.size >= fragment_limit) {
        auto block_it = free_fused_blocks.blocks.lower_bound(&p.search_key);
        if (block_it == free_fused_blocks.blocks.end() || (*block_it)->stream != p.stream() ||
            (*block_it)->size > (p.search_key.size * reuseLimit)) {
        } else {
            p.block = *block_it;
            activate_large_block(p.block);
            p.err = ACL_ERROR_NONE;

            update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
            update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);
            return true;
        }
    }

    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
        return false;
    }
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size())) {
        return false;
    }
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer)) {
        return false;
    }
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used

    if (pool.is_small) {
        pool.blocks.erase(p.block);
    }
    if (vmmDefragment > 0 && p.block->vmm_segment) {
        if (should_split(p.block, p.size())) {
            p.block = split_large_block(p.block, p.size());
        }
        activate_large_block(p.block);
    }

    p.err = ACL_ERROR_NONE;
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);
    return true;
}

bool DeviceCachingAllocator::trigger_free_memory_callbacks(AllocParams &p)
{
    bool freed_memory = false;
    return freed_memory;
}

void DeviceCachingAllocator::garbage_collect_cached_blocks()
{
    // Free unused cached blocks to reclaim NPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold =
        static_cast<size_t>(CachingAllocatorConfig::garbage_collection_threshold() * allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
        return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto &b : large_blocks.blocks) {
        if (!b->is_split()) {
            total_age += b->gc_count;
            ++freeable_block_count;
        }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
        return;
    }

    c10_npu::npuSynchronizeDevice(true);

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true && freeable_block_count > 0) {
        // Free blocks exceeding this age threshold first.
        double age_threshold = total_age / freeable_block_count;
        // Stop iteration if we can no longer free a block.
        block_freed = false;

        // Free blocks of > avg age. Don't stop upon reaching the target_size,
        // we don't want this GC to be triggered frequently.
        auto it = large_blocks.blocks.begin();
        while (it != large_blocks.blocks.end()) {
            Block *block = *it;
            ++it;
            if (!block->is_split() && block->gc_count >= age_threshold) {
                block_freed = true;
                gc_reclaimed += block->size;
                total_age -= block->gc_count; // Decrement the age
                freeable_block_count--;       // One less block that can be freed
                release_block(block);

                ASCEND_LOGD("PTA CachingAllocator gc: free = %zu, cached = %lu, allocated = %lu", block->size,
                    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
            }
        }
    }
}

bool DeviceCachingAllocator::realloc_block(AllocParams &p, bool isRetry)
{
    // Defensively checks for preexisting CUDA error state.
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    static const int reAlloc = ([]() -> int {
        const char *env = getenv("reAlloc");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    static const size_t fragment_limit = ([]() -> size_t {
        const char *env = getenv("fragLimit");
        if (env) {
            return static_cast<size_t>(std::stoll(env));
        } else {
            return static_cast<size_t>(16777216);
        }
    })();

    size_t size = p.alloc_size;
    size_t free_block_size = 0;
    void *ptr;

    if (isRetry) {
        stats.num_alloc_retries += 1;
    }

    std::shared_ptr<VmmSegment> vmm_segment;
    if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
        p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
        return false;
    }

    if (vmmDefragment <= 0 || p.pool->is_small) {
        p.err = aclrtMallocAlign32(&ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
        if (p.err != ACL_ERROR_NONE) {
            p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
            return false;
        }
        for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
            update_stat(stats.segment[stat_type], 1);
            update_stat(stats.reserved_bytes[stat_type], size);
        });
    } else {
        if (reAlloc > 0 && p.search_key.size > fragment_limit) {
            Block left_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool,
                p.search_key.ptr);
            Block right_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool,
                p.search_key.ptr);

            left_search_key.size = 0;
            right_search_key.size = std::numeric_limits<size_t>::max();

            auto it_begin = large_blocks.blocks.lower_bound(&left_search_key);
            auto it_end = large_blocks.blocks.lower_bound(&right_search_key);
            if (it_begin != large_blocks.blocks.end() && (*it_begin)->stream == p.stream() &&
                it_end != large_blocks.blocks.begin() && (*std::prev(it_end))->stream == p.stream()) {
                auto it = it_begin;
                while (it != it_end) {
                    free_block_size += (*it)->size;
                    it++;
                }
            }

            size_t request_size = p.search_key.size;
            if (free_block_size >= request_size) {
                return false;
            }

            if (free_block_size > 0) {
                request_size -= free_block_size;
                size = get_allocation_size(request_size);
            }
        }

        using Ms = std::chrono::duration<double, std::milli>;
        Ms fuse_time = Ms{ 0 };

        int gc_time = 0;
        do {
            auto t0 = std::chrono::steady_clock::now();

            vmm_segment = std::make_shared<VmmSegment>(size / kGranularity, kGranularity, p.device());

            auto t1 = std::chrono::steady_clock::now();
            fuse_time = (t1 - t0);

            if (vmm_segment->status == ACL_SUCCESS && vmm_segment->segment_ptr) {
                for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
                    update_stat(stats.segment[stat_type], 1);
                    update_stat(stats.reserved_bytes[stat_type], size);
                });
                break;
            } else {
                size_t device_free;
                size_t device_total;
                SWAP_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
                size_t total_garbage_size = fragmented_free_fused_blocks[p.stream()].pool_size +
                    free_fused_blocks_in_release_order[p.stream()].pool_size;
                if (device_free > size && total_garbage_size >= size) {
                    vmm_segment.reset();
                    size_t garbage_size = garbage_collect_fused_blocks(gc_time, p.alloc_size);
                    gc_time++;
                } else {
                    break;
                }
            }
        } while (gc_time < 3);

        if (!vmm_segment || vmm_segment->status != ACL_SUCCESS || !vmm_segment->segment_ptr) {
            p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
            vmm_segment.reset();
            return false;
        }
        ptr = vmm_segment->segment_ptr;
    }

    total_allocated_memory += size;
    Block *new_block = new Block(p.device(), p.stream(), size, p.pool, (char *)ptr);
    if (vmm_segment != nullptr) {
        new_block->vmm_segment = std::move(vmm_segment);
    }

    if (size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(new_block != nullptr && new_block->ptr != nullptr);

    if (new_block->vmm_segment) {
        if (new_block->size < p.search_key.size) {
            for (size_t i = 0; i < new_block->vmm_segment->phy_chunks.size(); i++) {
                new_block->vmm_segment->phy_chunks[i]->mapped_blocks.emplace_back(new_block, i);
                new_block->vmm_segment->phy_chunks[i]->free = true;
            }

            new_block->vmm_segment->num_free_chunks = new_block->vmm_segment->phy_chunks.size();
            new_block->vmm_segment->num_used_chunks = 0;

            large_blocks.blocks.insert(new_block);

            if (!get_fused_fragmented_blocks(p, 4)) {
                throw GMLakeError("Call get_fused_fragmented_blocks Failed");
            }
        } else {
            for (size_t i = 0; i < new_block->vmm_segment->phy_chunks.size(); i++) {
                new_block->vmm_segment->phy_chunks[i]->mapped_blocks.emplace_back(new_block, i);
                new_block->vmm_segment->phy_chunks[i]->free = false;
            }

            new_block->vmm_segment->num_free_chunks = 0;
            new_block->vmm_segment->num_used_chunks = new_block->vmm_segment->phy_chunks.size();

            p.block = new_block;
            p.err = ACL_ERROR_NONE;
        }
    } else {
        p.block = new_block;
        p.err = ACL_ERROR_NONE;
    }
    return true;
}

bool DeviceCachingAllocator::release_available_cached_blocks(const AllocParams &p)
{
    if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max()) {
        return false;
    }
    BlockPool &pool = *p.pool;
    Block key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
    key.size =
        (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
    auto it = pool.blocks.lower_bound(&key);

    c10_npu::npuSynchronizeDevice(true);

    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
        // No single block is large enough; free multiple oversize blocks, starting with the largest
        if (it == pool.blocks.begin()) {
            return false;
        }
        size_t totalReleased = 0;
        // Back up one item.  Now on the largest block for the correct stream
        --it;
        while ((totalReleased < key.size) && ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
            ((*it)->stream == p.stream())) {
            auto cur = it;
            totalReleased += (*it)->size;
            if (it != pool.blocks.begin()) {
                --it;
                release_block(*cur);
            } else {
                release_block(*cur);
                break;
            }
        }
        if (totalReleased < key.size) {
            return false;
        }
    } else {
        release_block(*it);
    }
    return true;
}

bool DeviceCachingAllocator::release_cached_blocks()
{
    c10_npu::npuSynchronizeDevice();
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();
    release_blocks(small_blocks);
    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);

    return true;
}

void DeviceCachingAllocator::release_block(Block *block)
{
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    if (block->pool->is_small || !block->vmm_segment->fused) {
        total_allocated_memory -= block->size;
        auto *pool = block->pool;
        StatTypes stat_types = { false };
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
            update_stat(stats.segment[stat_type], -1);
            update_stat(stats.reserved_bytes[stat_type], -static_cast<std::int64_t>(block->size));
        });
        if (block->size >= CachingAllocatorConfig::max_split_size())
            update_stat(stats.oversize_segments, -1);
    }

    if (vmmDefragment > 0 && block->vmm_segment) {
        release_large_block(block);
    } else {
        SWAP_CHECK_ERROR(aclrtFree(block->ptr));
        block->pool->blocks.erase(block);
        delete block;
    }
}

void DeviceCachingAllocator::release_blocks(BlockPool &pool)
{
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
        Block *block = *it;
        ++it;
        if (!block->prev && !block->next) {
            release_block(block);
        } else if (!block->pool->is_small && block->vmm_segment != nullptr) {
            if (block->prev && large_blocks.blocks.count(block->prev) == 1) {
                auto src = block->prev;
            }
            if (block->next && large_blocks.blocks.count(block->next) == 1) {
                auto src = block->next;
            }
        }
    }
}

EventPool::Event DeviceCachingAllocator::create_event_internal(int idx)
{
    // Leak the event pool to avoid shutdown issues.
    static auto *event_pool = new EventPool();
    return event_pool->get(idx);
}

void DeviceCachingAllocator::synchronize_and_free_events()
{
    // Synchronize on outstanding events and then free associated blocks.

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway == 0);
    insert_events_deferred_until_no_capture();

    for (auto &st : npu_events) {
        for (auto &e : st.second) {
            EventPool::Event event = std::move(e.first);
            Block *block = e.second;

            SWAP_CHECK_ERROR(aclrtSynchronizeEvent(*event));

            block->event_count--;
            if (block->event_count == 0) {
                update_block(block);
            }
        }
    }

    npu_events.clear();
}

void DeviceCachingAllocator::insert_events(Block *block)
{
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto &stream : streams) {
        c10_npu::SetDevice(stream.device_index());
        EventPool::Event event = create_event_internal(stream.device_index());
        event->record(stream);

        ASCEND_LOGI("Event: record DeviceAllocator is successfully executed.");
        block->event_count++;
        npu_events[stream].emplace_back(std::move(event), block);
    }
    if (ret_ctx == ACL_ERROR_NONE) {
        aclrtSetCurrentContext(compiler_ctx);
    }
}

void DeviceCachingAllocator::insert_free_event_into_alloc_stream(Block *block)
{
    int prev_device = -1;
    SWAP_CHECK_ERROR(c10_npu::GetDevice(&prev_device));
    if (prev_device != block->device) {
        SWAP_CHECK_ERROR(c10_npu::SetDevice(block->device));
    }

    if (prev_device != block->device) {
        SWAP_CHECK_ERROR(c10_npu::SetDevice(prev_device));
    }
}

void DeviceCachingAllocator::insert_events_deferred_until_no_capture()
{
    if (C10_UNLIKELY(needs_events_deferred_until_no_capture.size() > 0)) {
        for (auto *block : needs_events_deferred_until_no_capture) {
            TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
            insert_events(block);
        }
        needs_events_deferred_until_no_capture.clear();
    }
}

void DeviceCachingAllocator::process_events()
{
    // Process outstanding npuEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    for (auto it = npu_events.begin(); it != npu_events.end();) {
        while (!it->second.empty()) {
            auto &e = it->second.front();
            EventPool::Event event = std::move(e.first);
            Block *block = e.second;

            if (!event->query()) {
                e.first = std::move(event);
                break;
            }

            block->event_count--;
            if (block->event_count == 0) {
                update_block(block);
            }
            it->second.pop_front();
        }

        if (it->second.empty()) {
            it = npu_events.erase(it);
        } else {
            it++;
        }
    }
}

void DeviceCachingAllocator::cache_info_aux(BlockPool &blocks, size_t *total, size_t *largest)
{
    for (auto it = blocks.blocks.begin(); it != blocks.blocks.end(); ++it) {
        size_t blocksize = (*it)->size;
        *total += blocksize;
        if (blocksize > *largest) {
            *largest = blocksize;
        }
    }
}

bool DeviceCachingAllocator::get_fused_fragmented_blocks(AllocParams &p, int time)
{
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    static const size_t fragment_limit = ([]() -> size_t {
        const char *env = getenv("fragLimit");
        if (env) {
            return static_cast<size_t>(std::stoll(env));
        } else {
            return static_cast<size_t>(16777216);
        }
    })();

    static const int defragment_level = ([]() -> int {
        const char *env = getenv("defragLevel");
        if (env) {
            return static_cast<int>(std::atoi(env));
        } else {
            return 0;
        }
    })();

    static const int auto_gc_limits = ([]() -> int {
        const char *env = getenv("autoGC");
        if (env) {
            return static_cast<int>(std::atoi(env));
        } else {
            return 3000;
        }
    })();

    static const int split_limit = ([]() -> int {
        const char *env = getenv("split_limit");
        if (env) {
            return static_cast<int>(std::atoi(env));
        } else {
            return 10;
        }
    })();

    if (vmmDefragment <= 0) {
        return false;
    }

    if (time < defragment_level) {
        return false;
    }

    if (p.pool->is_small || p.search_key.size < fragment_limit) {
        return false;
    }

    Block left_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool,
        p.search_key.ptr);
    Block right_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool,
        p.search_key.ptr);

    left_search_key.size = 0;
    right_search_key.size = std::numeric_limits<size_t>::max();

    auto it_begin = large_blocks.blocks.lower_bound(&left_search_key);
    if (it_begin == large_blocks.blocks.end() || (*it_begin)->stream != p.stream()) {
        return false;
    }
    auto it_end = large_blocks.blocks.lower_bound(&right_search_key);
    if (it_end == large_blocks.blocks.begin() || (*std::prev(it_end))->stream != p.stream()) {
        return false;
    }

    if (std::prev(it_end) == it_begin) {
        return false;
    }

    size_t fuse_size = 0;
    std::vector<Block *> blocks2fuse;

    auto it = it_end;
    while (it != it_begin) {
        it = std::prev(it);
        if (fuse_size + (*it)->size >= p.search_key.size) {
            Block last_block_search_key(p.search_key.device, p.search_key.stream, p.search_key.size - fuse_size,
                p.search_key.pool, p.search_key.ptr);
            auto last_block_it = large_blocks.blocks.lower_bound(&last_block_search_key);
            blocks2fuse.push_back((*last_block_it));
            fuse_size += (*last_block_it)->size;
            break;
        } else {
            blocks2fuse.push_back((*it));
            fuse_size += (*it)->size;
        }
    }

    if (fuse_size < p.search_key.size) {
        return false;
    }

    if (fuse_size > p.search_key.size && (fuse_size - p.search_key.size) >= kGranularity) {
        Block *last_block = blocks2fuse.back();
        blocks2fuse.pop_back();
        size_t original_size = last_block->size;
        size_t remain_size = (fuse_size - p.search_key.size);
        size_t keep_size = original_size - remain_size;
        Block *a = split_large_block(last_block, keep_size);
        blocks2fuse.push_back(a);
    }

    int64_t net_change_segments = 0;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    std::vector<std::shared_ptr<PhyBlock>> phy_chunks2glue;
    auto sblock = stitch_block(blocks2fuse, p);
    activate_large_block(sblock);
    p.block = sblock;
    p.err = ACL_ERROR_NONE;

    net_change_segments += 1;

    update_stat_array(stats.segment, net_change_segments, p.stat_types);
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);

    return fuse_size >= p.search_key.size;
}

bool DeviceCachingAllocator::release_swapout_blocks()
{
    return c10_npu::swap::NPUSwapManager::GetInstance().ProcessMallocEvent();
}

Block *DeviceCachingAllocator::stitch_block(std::vector<Block *> &blocks2fuse, AllocParams &p)
{
    static constexpr size_t G = 1024 * 1024 * 1024;
    static const int auto_gc_limits = ([]() -> int {
        const char *env = getenv("autoGC");
        if (env) {
            return static_cast<int>(std::atoi(env));
        } else {
            return 3000;
        }
    })();

    std::vector<std::shared_ptr<PhyBlock>> phy_chunks2glue;

    for (auto &block : blocks2fuse) {
        for (auto &phy_block : block->vmm_segment->phy_chunks) {
            phy_chunks2glue.push_back(phy_block);
        }
    }
    size_t fuse_size = phy_chunks2glue.size() * kGranularity;
    using Ms = std::chrono::duration<double, std::milli>;
    Ms fuse_time = Ms{ 0 };
    std::shared_ptr<VmmSegment> vmm_segment;
    int gc_time = 0;
    do {
        auto t0 = std::chrono::steady_clock::now();
        vmm_segment = std::make_shared<VmmSegment>(std::move(phy_chunks2glue));
        auto t1 = std::chrono::steady_clock::now();
        fuse_time = (t1 - t0);
        if (vmm_segment->status == ACL_SUCCESS && vmm_segment->segment_ptr) {
            break;
        } else {
            phy_chunks2glue = std::move(vmm_segment->phy_chunks);
            size_t garbage_size = garbage_collect_fused_blocks(gc_time, fuse_size);
            gc_time++;
        }
    } while (gc_time < 3);

    if (!vmm_segment || vmm_segment->status != ACL_SUCCESS || !vmm_segment->segment_ptr) {
        throw GMLakeError("stitch pBlocks failed, something wrong happended !");
    }

    void *block_ptr = vmm_segment->segment_ptr;
    Block *fused_block = new Block(p.device(), p.stream(), fuse_size, p.pool, (char *)block_ptr);
    fused_block->vmm_segment = std::move(vmm_segment);
    size_t offset = 0;
    for (auto &phy_block : fused_block->vmm_segment->phy_chunks) {
        phy_block->mapped_blocks.emplace_back(fused_block, offset);
        offset++;
    }
    fused_block->vmm_segment->num_free_chunks = fused_block->vmm_segment->phy_chunks.size();
    fused_block->vmm_segment->num_used_chunks = 0;

    total_fuse_size += fuse_size;
    if (total_fuse_size > auto_gc_limits * G) {
        size_t garbage_size = garbage_collect_fused_blocks(2, 0);
    }
    free_fused_blocks.blocks.insert(fused_block);
    free_fused_blocks.hash.insert(fused_block->ptr_hash);
    return fused_block;
}

Block *DeviceCachingAllocator::split_large_block(Block *block, size_t request_size)
{
    static const int vmmDefragment = ([]() -> int {
        const char *env = getenv("vmmDefragment");
        if (env) {
            return atoi(env);
        } else {
            return 1;
        }
    })();

    large_blocks.blocks.erase(block);

    const bool already_split = block->is_split();
    const bool is_block_free = large_blocks.blocks.count(block) == 1 ? true : false;

    Block *remaining_block = block;
    block = new Block(block->device, block->stream, request_size, block->pool, block->ptr);
    block->prev = remaining_block->prev;
    if (block->prev) {
        block->prev->next = block;
    }
    block->next = remaining_block;

    remaining_block->prev = block;
    remaining_block->ptr = static_cast<char *>(remaining_block->ptr) + request_size;
    remaining_block->size -= request_size;

    if (vmmDefragment > 0 && remaining_block->vmm_segment) {
        auto remaining_segment = remaining_block->vmm_segment->split(request_size);
        block->vmm_segment = std::move(remaining_block->vmm_segment);
        remaining_block->vmm_segment = std::move(remaining_segment);

        size_t offset = 0;
        for (auto &phy_block : block->vmm_segment->phy_chunks) {
            phy_block->mapped_blocks[0].block = block;
            phy_block->mapped_blocks[0].offset = offset;
            phy_block->free = true;
            offset++;
        }

        block->vmm_segment->num_free_chunks = block->vmm_segment->phy_chunks.size();
        block->vmm_segment->num_used_chunks = 0;

        offset = 0;
        for (auto &phy_block : remaining_block->vmm_segment->phy_chunks) {
            phy_block->mapped_blocks[0].block = remaining_block;
            phy_block->mapped_blocks[0].offset = offset;
            phy_block->free = true;
            offset++;
        }
        remaining_block->vmm_segment->num_free_chunks = remaining_block->vmm_segment->phy_chunks.size();
        remaining_block->vmm_segment->num_used_chunks = 0;
    }

    large_blocks.blocks.insert(block);
    large_blocks.blocks.insert(remaining_block);
    remaining_block->allocated = false;
    block->allocated = false;
    return block;
}

void DeviceCachingAllocator::release_large_block(Block *block)
{
    if (!block->vmm_segment->fused) {
        // 确认pblock内所有chunk关联的pblock/sblock是否一致
        // sblock集合，存储待释放的sblock
        // 抽象为release_pblock release_sblock
        for (auto &phy_block : block->vmm_segment->phy_chunks) {
            while (phy_block->mapped_blocks.size() > 1) {
                release_large_block(phy_block->mapped_blocks[1].block);
            }
        }
    }
    if (block->vmm_segment->fused) {
        total_fuse_size -= block->size;
    }

    if (free_fused_blocks.hash.count(block->ptr_hash)) {
        free_fused_blocks.blocks.erase(block);
        free_fused_blocks.hash.erase(block->ptr_hash);
    } else if (fragmented_free_fused_blocks[block->stream].blocks.count(block)) {
        fragmented_free_fused_blocks[block->stream].erase(block);
    } else if (large_blocks.blocks.count(block)) {
        large_blocks.blocks.erase(block);
    }
    for (auto &phy_block : block->vmm_segment->phy_chunks) {
        int i = 0;
        for (int j = 0; j < phy_block->mapped_blocks.size(); j++) {
            if (phy_block->mapped_blocks[j].block != block) {
                if (i != j) {
                    phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                }
                i++;
            }
        }
        phy_block->mapped_blocks.resize(i);
    }

    {
        auto tmp = std::move(block->vmm_segment);
    }
    delete block;
}

void DeviceCachingAllocator::activate_large_block(Block *block)
{
    ska::flat_hash_set<Block *> active_pblocks;
    if (block->vmm_segment->fused) {
        free_fused_blocks.blocks.erase(block);
        free_fused_blocks.hash.erase(block->ptr_hash);
        active_fused_blocks.insert(block);
    } else {
        large_blocks.blocks.erase(block);
        active_blocks.insert(block);
    }
    int phy_chunks_size = block->vmm_segment->phy_chunks.size();
    int vir_chunks_size = block->vmm_segment->vir_chunks.size();

    for (int i = 0; i < phy_chunks_size; i++) {
        auto chunk = block->vmm_segment->phy_chunks[i];
        chunk->free = false;
        block->vmm_segment->num_free_chunks--;
        block->vmm_segment->num_used_chunks++;

        for (int j = 0; j < chunk->mapped_blocks.size(); j++) {
            Block *other_block = chunk->mapped_blocks[j].block;
            if (other_block == block) {
                continue;
            }
            if (other_block->vmm_segment->fused) {
                if (free_fused_blocks.hash.count(other_block->ptr_hash) == 1) {
                    free_fused_blocks.blocks.erase(other_block);
                    free_fused_blocks.hash.erase(other_block->ptr_hash);
                    fragmented_free_fused_blocks[other_block->stream].insert(other_block);
                }
            } else {
                if (large_blocks.blocks.count(other_block) == 1) {
                    large_blocks.blocks.erase(other_block);
                    other_block->allocated = true;
                    active_blocks.insert(other_block);
                    active_pblocks.insert(other_block);
                }
            }
            other_block->vmm_segment->num_free_chunks--;
            other_block->vmm_segment->num_used_chunks++;
        }
    }
}

void DeviceCachingAllocator::deactivate_large_block(Block *block)
{
    ska::flat_hash_set<Block *> active_pblocks;
    if (block->vmm_segment->fused) {
        active_fused_blocks.erase(block);
        free_fused_blocks.blocks.insert(block);
        free_fused_blocks.hash.insert(block->ptr_hash);
    } else {
        active_blocks.erase(block);
    }
    int phy_chunks_size = block->vmm_segment->phy_chunks.size();
    int vir_chunks_size = block->vmm_segment->vir_chunks.size();
    AT_ASSERT(phy_chunks_size == vir_chunks_size,
        "when inactive_block, phy_chunks_size is not equal to vir_chunks_size");
    for (int i = 0; i < phy_chunks_size; i++) {
        auto chunk = block->vmm_segment->phy_chunks[i];
        chunk->free = true;
        block->vmm_segment->num_used_chunks--;
        block->vmm_segment->num_free_chunks++;

        for (int j = 0; j < chunk->mapped_blocks.size(); j++) {
            Block *other_block = chunk->mapped_blocks[j].block;
            if (other_block == block) {
                continue;
            }
            if (other_block->vmm_segment->fused) {
                other_block->vmm_segment->num_free_chunks++;
                other_block->vmm_segment->num_used_chunks--;
                if (other_block->vmm_segment->num_used_chunks == 0) {
                    fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                    free_fused_blocks.blocks.insert(other_block);
                    free_fused_blocks.hash.insert(other_block->ptr_hash);
                }
            } else {
                if (active_blocks.count(other_block) == 1) {
                    other_block->allocated = false;
                    active_pblocks.insert(other_block);
                }

                other_block->vmm_segment->num_free_chunks++;
                other_block->vmm_segment->num_used_chunks--;
            }
        }
    }
    if (!block->vmm_segment->fused) {
        TORCH_INTERNAL_ASSERT(!block->allocated && block->event_count == 0 && block->stream_uses.empty());

        auto &pool = *block->pool;
        const std::array<Block *, 2> merge_candidates = { block->prev, block->next };
        for (Block *merge_candidate : merge_candidates) {
            try_merge_blocks(block, merge_candidate, pool);
        }
        large_blocks.blocks.insert(block);
    }

    for (auto &other_block : active_pblocks) {
        free_block(other_block, false);
    }
}

size_t DeviceCachingAllocator::garbage_collect_fused_blocks(int time, size_t require_size)
{
    c10_npu::npuSynchronizeDevice(true);

    static const int gc_thresh = ([]() -> int {
        const char *env = getenv("gc_thresh");
        if (env) {
            return atoi(env);
        } else {
            return 100;
        }
    })();

    std::lock_guard<std::recursive_mutex> lock(mutex);

    size_t garbage_size = 0;
    size_t garbage_blocks = 0;

    const size_t G = 1024 * 1024 * 1024;
    for (auto &it : fragmented_free_fused_blocks) {
        while (!it.second.blocks.empty() && garbage_size <= gc_thresh * G) {
            Block *block = *(it.second.blocks.begin());
            aclError err = ACL_ERROR_NONE;
            aclrtEventRecordedStatus eventStatus = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            if (err == ACL_ERROR_NONE) {
                garbage_blocks++;
                garbage_size += block->size;
                release_large_block(block);
            } else {
                break;
            }
        }
    }

    if (time > 0) {
        while (!free_fused_blocks.blocks.empty()) {
            Block *block = *(free_fused_blocks.blocks.begin());
            garbage_size += block->size;
            release_large_block(block);
            if (garbage_size <= gc_thresh * G) {
                break;
            }
        }
    }

    return garbage_size;
}