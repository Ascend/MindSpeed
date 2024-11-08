#include "DeviceCachingAllocator.h"
#include <torch/csrc/python_headers.h>

DeviceCachingAllocator::DeviceCachingAllocator()
    : long_lc_pools(BLOCK_POOL_LONG), default_lc_pools(BLOCK_POOL_DEFAULT) {
    stats.max_split_size = static_cast<int64_t>(CachingAllocatorConfig::max_split_size());
}

void DeviceCachingAllocator::print_memory_analysis() {
    std::vector<SegmentInfo> seg_array = snapshot();
    std::map<std::string, std::pair<size_t, size_t>> memory_cnt;
    for (SegmentInfo &seg : seg_array) {
        std::string seg_type = seg.is_large ? "large" : "small";
#ifdef MEMORY_RECORDER_DEBUG
        seg_type += seg.type_str;
#endif
        printf("SEG info: dev %ld, size %lu, allocated %lu, type %s\n", seg.device, seg.total_size, seg.allocated_size,
               seg_type.c_str());
        std::vector<size_t> active, inactive, allocated, notallocated;
        for (BlockInfo& blk : seg.blocks) {
            if (blk.active) {
                active.push_back(blk.size);
            } else {
                inactive.push_back(blk.size);
            }
            if (blk.allocated) {
                allocated.push_back(blk.size);
            } else {
                notallocated.push_back(blk.size);
            }
        }
        size_t active_cnt = std::accumulate(active.begin(), active.end(), (size_t)0);
        size_t inactive_cnt = std::accumulate(inactive.begin(), inactive.end(), (size_t)0);
        size_t allocated_cnt = std::accumulate(allocated.begin(), allocated.end(), (size_t)0);
        size_t notallocated_cnt = std::accumulate(notallocated.begin(), notallocated.end(), (size_t)0);

        auto& active_info = memory_cnt[seg_type + "-active"];
        auto& inactive_info = memory_cnt[seg_type + "-inactive"];
        auto& allocated_info = memory_cnt[seg_type + "-allocated"];
        auto& notallocated_info = memory_cnt[seg_type + "-notallocated"];

        active_info.first += active.size();
        active_info.second += active_cnt;
        inactive_info.first += inactive.size();
        inactive_info.second += inactive_cnt;

        allocated_info.first += allocated.size();
        allocated_info.second += allocated_cnt;
        notallocated_info.first += notallocated.size();
        notallocated_info.second += notallocated_cnt;
    }
    for (auto &x : memory_cnt) {
        printf("%s cnt %lu size %lu\n", x.first.c_str(), x.second.first, x.second.second);
    }
    TORCH_CHECK(fflush(stdout) == 0, "fflush failed.");
}

Block* DeviceCachingAllocator::malloc(int device, size_t orig_size, aclrtStream stream) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (device == -1) {
        TORCH_CHECK(c10_npu::GetDevice(&device) == ACL_ERROR_NONE, "GetDevice failed.");
    }

    size_t tensor_forward_end = std::numeric_limits<size_t>::max();
    size_t tensor_forward_start = recorder.forward_tik;
    // Obtain the lifecycle of the current tensor
    LifeCycleType lc = recorder.get_lc(orig_size, &tensor_forward_end, &tensor_forward_start);
    // Call the current malloc_internal to apply for a block
    Block *ret = malloc_internal(device, orig_size, stream, lc, tensor_forward_end, tensor_forward_start);
    return ret;
}

Block* DeviceCachingAllocator::malloc_internal(int device, size_t orig_size, aclrtStream stream, LifeCycleType lc,
                                               size_t tensor_forward_end, size_t tensor_forward_start) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (device == -1) {
        TORCH_CHECK(c10_npu::GetDevice(&device) == ACL_ERROR_NONE, "GetDevice failed.");
    }

    // process outstanding npuEvents
    process_events();
    auto size = round_size(orig_size);
    size_t tensor_step_end = std::numeric_limits<size_t>::max();
    size_t tensor_step_start = malloc_recorder.tik;
    // Determine if a tensor has a long lifecycle
    if (!is_precise_match && malloc_recorder.predict_long(size, &tensor_step_end, &tensor_step_start)) {
        lc = LifeCycleType::LONG_LC;
    }

    size_t default_lc_threshold =
            static_cast<size_t>(static_cast<long long>(CachingAllocatorConfig::default_lc_threshold()));
    if (size <= default_lc_threshold) lc = LifeCycleType::DEFAULT_LC;

    auto pool_list = get_pool_list(size, lc);
    size_t alloc_size = 0;
    AllocParams params(device, size, stream, pool_list[0], alloc_size, stats);
    bool block_found = false;

    while (!block_found) {
        BlockPool* pool;
        pool_idx = 0;

        for (auto iter_pool : pool_list) {
            pool = iter_pool;
            alloc_size = get_allocation_size(
                    size, pool == &long_lc_pools.large_blocks ? LifeCycleType::LONG_LC : LifeCycleType::DEFAULT_LC);
            AllocParams _params(device, size, stream, pool, alloc_size, stats);
            _params.stat_types = get_stat_types_for_pool(*pool);

            if (CachingAllocatorConfig::open_memory_optimize()) { // When the tensor lifecycle conflicts
                block_found =
                        // Search pool
                        get_free_block_memory_optimize(_params, tensor_forward_end, tensor_step_end, tensor_forward_start,
                                                       tensor_step_start)
                        // Trigger callbacks and retry search
                        || (trigger_free_memory_callbacks(_params) &&
                            get_free_block_memory_optimize(_params, tensor_forward_end, tensor_step_end, tensor_forward_start,
                                                           tensor_step_start));
            } else {
                block_found =
                        // Search pool
                        get_free_block(_params)
                        // Trigger callbacks and retry search
                        || (trigger_free_memory_callbacks(_params) && get_free_block(_params));
            }

            params = _params;

            if (block_found) {
                break;
            }
            pool_idx++;
        }

        if (!block_found) {
            pool = pool_list[0];
            alloc_size = get_allocation_size(
                    size, pool == &long_lc_pools.large_blocks ? LifeCycleType::LONG_LC : LifeCycleType::DEFAULT_LC);
            AllocParams _params(device, size, stream, pool, alloc_size, stats);
            _params.stat_types = get_stat_types_for_pool(*pool);

            block_found =
                    // Attempt allocate
                    alloc_block(_params, false) ||
                    // Free enough available cached blocks to satisfy alloc and retry alloc.
                    (release_available_cached_blocks(_params) && alloc_block(_params, false));
            params = _params;
        }

        pool_idx = 0;
        if (!block_found) {
            // Prioritize searching in another pool
            for (auto pool_it = pool_list.rbegin(); pool_it != pool_list.rend(); ++pool_it) {
                pool = *pool_it;
                alloc_size = get_allocation_size(
                    size, pool == &long_lc_pools.large_blocks ? LifeCycleType::LONG_LC : LifeCycleType::DEFAULT_LC);
                AllocParams _params(device, size, stream, pool, alloc_size, stats);
                _params.stat_types = get_stat_types_for_pool(*pool);

                block_found =
                        // Search pool
                        get_free_block_after_alloc(_params)
                        // Trigger callbacks and retry search
                        || (trigger_free_memory_callbacks(_params) && get_free_block_after_alloc(_params));

                params = _params;

                if (block_found) {
                    break;
                }
                pool_idx++;
            }
        }

        // When it is a small tensor, search in a large memory pool to prevent OOM
        if (!block_found && size <= kSmallSize) {
            pool_list = get_pool_list(kLargeBuffer, lc);
            for (auto pool_it = pool_list.begin(); pool_it != pool_list.end(); ++pool_it) {
                pool = *pool_it;
                alloc_size = get_allocation_size(
                    size, pool == &long_lc_pools.large_blocks ? LifeCycleType::LONG_LC : LifeCycleType::DEFAULT_LC);
                AllocParams _params(device, size, stream, pool, alloc_size, stats);
                _params.stat_types = get_stat_types_for_pool(*pool);

                block_found =
                        // Search pool
                        get_free_block_after_alloc(_params)
                        // Trigger callbacks and retry search
                        || (trigger_free_memory_callbacks(_params) && get_free_block_after_alloc(_params));

                params = _params;

                if (block_found) {
                    break;
                }
                pool_idx++;
            }
        }

        if (!block_found) {
            pool = pool_list[0];
            alloc_size = get_allocation_size(
                    size, pool == &long_lc_pools.large_blocks ? LifeCycleType::LONG_LC : LifeCycleType::DEFAULT_LC);

            if (pool == &long_lc_pools.large_blocks || pool == &long_lc_pools.small_blocks) {
                printf("try long_lc pool fail, size:%lu\n", alloc_size);
            } else {
                printf("try default_lc pool fail, size:%lu\n", alloc_size);
            }
            AllocParams _params(device, size, stream, pool, alloc_size, stats);
            _params.stat_types = get_stat_types_for_pool(*pool);
            block_found = release_cached_blocks_default(true) && release_cached_blocks_long(true) && alloc_block(_params, true);
            params = _params;
        }

        if (!block_found) {
            if (params.err == ACL_ERROR_NONE) {
                break;
            }
            PyGILState_STATE state = PyGILState_Ensure();
            PyObject *pModule = PyImport_ImportModule("mindspeed.core.memory.adaptive_recomputing.swap_manager");
            if (!pModule) {
                std::cout << "No MindSpeed Module" << std::endl;
                PyGILState_Release(state);
                break;
            }
            PyObject *pFunc1 = PyObject_GetAttrString(pModule, "SwapManager");
            PyObject *pClass = PyObject_CallObject(pFunc1, nullptr);
            PyObject *pFunc2 = PyObject_GetAttrString(pClass, "swap_out_by_size");

            PyObject *pArgs = PyTuple_New(1);
            TORCH_CHECK(PyTuple_SetItem(pArgs, 0, PyLong_FromLong(size)) == 0, "PyTuple_SetItem failed.");

            PyObject *pResult = PyObject_CallObject(pFunc2, pArgs);
            bool ret = false;
            TORCH_CHECK(PyArg_Parse(pResult, "p", &ret), "PyArg_Parse failed.");
            PyGILState_Release(state);
            if (!ret) {
                std::cout << "SWAP Failed" << std::endl;
                break;
            }
            params.err = ACL_ERROR_NONE;
        }
    }

    if (!block_found) {
        if (params.err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
            size_t device_free;
            size_t device_total;
            TORCH_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total) == ACL_ERROR_NONE, "aclrtGetMemInfo failed.");
            TORCH_INTERNAL_ASSERT(device_free <= device_total);

            std::string allowed_info;
            if (set_fraction) {
                allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
            }
            stats.num_ooms += 1;
            print_memory_analysis();
            // "total capacity": total global memory on NPU
            // "allowed": memory is allowed to use, which set by fraction.
            // "already allocated": memory allocated by the program using the
            //                      caching allocator
            // "free": free memory as reported by the NPU API
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
        }
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(std::move(params), orig_size, split_remainder);
}

Block* DeviceCachingAllocator::alloc_found_block(AllocParams params, size_t orig_size, bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(params.err == ACL_ERROR_NONE && params.block != nullptr && params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
        remaining = block;

        block = new Block(device, stream, size, pool, block->ptr);
        block->expandable_segment_ = remaining->expandable_segment_;
        block->prev = remaining->prev;
        if (block->prev) {
            block->prev->next = block;
        }
        block->next = remaining;

        remaining->prev = block;
        remaining->ptr = static_cast<char*>(remaining->ptr) + size;
        remaining->size -= size;
        pool->blocks.insert(remaining);

        if (already_split && !block->expandable_segment_) {
            // An already-split inactive block is being shrunk by size bytes.
            update_stat_array(stats.inactive_split_bytes,-static_cast<std::int64_t>(block->size), params.stat_types);
        } else if (!block->expandable_segment_) {
            // A new split inactive block is being created from a previously unsplit
            // block, size remaining->size bytes.
            for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                update_stat(stats.inactive_split_bytes[stat_type], static_cast<std::int64_t>(remaining->size));
                update_stat(stats.inactive_split[stat_type], 1);
            });
        }
    } else if (already_split && !block->expandable_segment_) {
        // An already-split block is becoming active
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
            update_stat(stats.inactive_split_bytes[stat_type], -static_cast<std::int64_t>(block->size));
            update_stat(stats.inactive_split[stat_type], -1);
        });
    }

    block->allocated = true;
    block->requested_size = orig_size;

    active_blocks.insert(block);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.allocation[stat_type], 1);
        update_stat(stats.allocated_bytes[stat_type], static_cast<std::int64_t>(block->size));
        update_stat(stats.active[stat_type], 1);
        update_stat(stats.active_bytes[stat_type], static_cast<std::int64_t>(block->size));
        update_stat(stats.requested_bytes[stat_type], static_cast<std::int64_t>(block->requested_size));
    });

    if (block->size >= CachingAllocatorConfig::max_split_size()) update_stat(stats.oversize_allocations, 1);

    ASCEND_LOGD("PTA CachingAllocator malloc: malloc = %zu, cached = %lu, allocated = %lu", block->size,
                stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);

    c10::reportMemoryUsageToProfiler(block->ptr, block->size,
                                     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                                     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                                     c10::Device(c10::DeviceType::PrivateUse1, block->device));

    // Record the forward information of the tensor
    block->forward_start_tik = recorder.forward_tik++;
    recorder.add(block->forward_start_tik, std::numeric_limits<size_t>::max(), orig_size);
    block->orig_size = orig_size;
    block->forward_count = recorder.forward_count;
    block->in_forward = _check();

    // Record the step information of the tensor
    block->start_tik = malloc_recorder.tik++;
    malloc_recorder.add(block->start_tik, std::numeric_limits<size_t>::max(), size);
    block->tensor_size = size;
    block->step_count = malloc_recorder.step_count;
    block->in_step = malloc_recorder._check();

    return block;
}

void DeviceCachingAllocator::free(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // Tensor information processing in the step stage
    TORCH_INTERNAL_ASSERT(malloc_recorder.step_count >= block->step_count);
    unsigned int step_distance = malloc_recorder.step_count - block->step_count;
    malloc_recorder.change_end_tik(block->start_tik, malloc_recorder.tik, block->tensor_size, step_distance,
                                   block->in_step);

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    // Tensor information processing in the forward stage
    TORCH_INTERNAL_ASSERT(recorder.forward_count >= block->forward_count);
    unsigned int forward_distance = recorder.forward_count - block->forward_count;
    recorder.change_forward_end_tik(block->forward_start_tik, recorder.forward_tik, block->orig_size, forward_distance,
                                    block->in_forward);

    StatTypes stat_types = get_stat_types_for_pool(*(block->pool));
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.allocation[stat_type], -1);
        update_stat(stats.allocated_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size()) update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty() && !shutdown_stats) {
        insert_events(block);
    } else {
        free_block(block);
    }

    ASCEND_LOGD("PTA CachingAllocator free: free = %zu, cached = %lu, allocated = %lu", orig_block_size,
                stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);

    c10::reportMemoryUsageToProfiler(orig_block_ptr, -orig_block_size,
                                     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                                     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                                     c10::Device(c10::DeviceType::PrivateUse1, block->device));
}

void* DeviceCachingAllocator::get_base_allocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
        block = block->prev;
    }
    void* basePtr = block->ptr;
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

void DeviceCachingAllocator::record_stream(Block* block, c10_npu::NPUStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    block->stream_uses.insert(stream);
}

void DeviceCachingAllocator::erase_stream(Block* block, c10_npu::NPUStream stream) {
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
            free_block(block);
            break;
        }
    }
}

void DeviceCachingAllocator::set_memory_fraction(double fraction) {
    size_t device_free;
    size_t device_total;
    TORCH_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total) == ACL_ERROR_NONE, "acl interface call failed");
    TORCH_INTERNAL_ASSERT(device_free <= device_total);
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
}

void DeviceCachingAllocator::empty_cache(bool check_error) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(release_cached_blocks(check_error), "release_cached_blocks failed.");
}

void DeviceCachingAllocator::dev_set_shutdown_stats() { shutdown_stats = true; }

void DeviceCachingAllocator::cache_info(size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cache_info_aux(default_lc_pools.large_blocks, total, largest);
    cache_info_aux(default_lc_pools.small_blocks, total, largest);
    cache_info_aux(long_lc_pools.large_blocks, total, largest);
    cache_info_aux(long_lc_pools.small_blocks, total, largest);
}

DeviceStats DeviceCachingAllocator::get_stats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
}

void DeviceCachingAllocator::reset_accumulated_stats() {
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

void DeviceCachingAllocator::reset_peak_stats() {
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

std::vector<SegmentInfo> DeviceCachingAllocator::snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
        // For expandable segments, we report one segment for each continguous
        // mapped range of memory
        if (head_block->prev && head_block->prev->mapped) {
            continue;
        }
        result.emplace_back();
        SegmentInfo& segment_info = result.back();
        segment_info.device = head_block->device;
        segment_info.address = reinterpret_cast<uintptr_t>(head_block->ptr);
        segment_info.is_large = (!head_block->pool->is_small);
        segment_info.is_expandable = head_block->expandable_segment_;
#ifdef MEMORY_RECORDER_DEBUG
        segment_info.type = head_block->pool->type;
        segment_info.type_str = get_block_pool_str(segment_info.type);
#endif

        const Block* block = head_block;
        while (block != nullptr && block->mapped) {
            segment_info.blocks.emplace_back();
            BlockInfo& block_info = segment_info.blocks.back();

            block_info.size = block->size;
            block_info.allocated = block->allocated;
            block_info.active = block->allocated || (block->event_count > 0);

            segment_info.total_size += block_info.size;
            if (block_info.allocated) {
                segment_info.allocated_size += block_info.size;
            }
            if (block_info.active) {
                segment_info.active_size += block_info.size;
            }

            block = block->next;
        }
    }

    std::sort(result.begin(), result.end(),
              [](const SegmentInfo& a, const SegmentInfo& b) { return a.address < b.address; });

    return result;
}

size_t DeviceCachingAllocator::round_size(size_t size) {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
}

std::vector<const Block*> DeviceCachingAllocator::get_all_blocks() const {
    std::vector<const Block *> blocks;
    const BlockPool* pools[] = {
            &long_lc_pools.small_blocks,
            &long_lc_pools.large_blocks,
            &default_lc_pools.small_blocks,
            &default_lc_pools.large_blocks
    };
    for (auto pool : pools) {
        blocks.insert(blocks.end(), pool->blocks.begin(), pool->blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
}

Block* DeviceCachingAllocator::find_expandable_block(int device, aclrtStream stream, BlockPool* pool, size_t size) {
    Block key(device, stream, 0);

    auto allocatable = [](Block* b) { return b && !b->allocated && b->event_count == 0 && b->stream_uses.empty(); };
    auto has_available_address_space = [&](Block* b) {
        size_t bytes = 0;
        while (bytes < size && allocatable(b)) {
            bytes += b->size;
            b = b->next;
        }
        return bytes >= size;
    };
    for (auto it = pool->unmapped.lower_bound(&key); it != pool->unmapped.end() && (*it)->stream == stream; ++it) {
        Block* c = *it;
        // we found the lowest address of an unmapped segment
        // but there might be a free segment we can also use
        // right before it
        if (allocatable(c->prev)) {
            c = c->prev;
        }
        if (has_available_address_space(c)) {
            return c;
        }
    }
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    expandable_segments_.emplace_back(new ExpandableSegment(device, stream, segment_size));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
}

bool DeviceCachingAllocator::map_block(Block* to_map, size_t size) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    auto mapped_range = to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
    if (mapped_range.size == 0) {
        return false;
    }
    TORCH_INTERNAL_ASSERT(mapped_range.ptr == to_map->ptr && mapped_range.size >= size);

    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
        // to_map -> remaining -> to_map->next(?)
        Block* remaining = new Block(to_map->device, to_map->stream, to_map->size - mapped_range.size, &pool,
                static_cast<char*>(to_map->ptr) + mapped_range.size);
        remaining->mapped = false;
        remaining->expandable_segment_ = to_map->expandable_segment_;
        remaining->splice(to_map, to_map->next);
        pool.unmapped.insert(remaining);
        to_map->size = mapped_range.size;
    }

    TORCH_CHECK(try_merge_blocks(to_map, to_map->prev, pool) >= 0, "try_merge_blocks failed.");
    TORCH_CHECK(try_merge_blocks(to_map, to_map->next, pool) >= 0, "try_merge_blocks failed.");

    pool.blocks.insert(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(
            stat_types, [&](size_t stat_type) { update_stat(stats.reserved_bytes[stat_type], mapped_range.size); });

    return true;
}

Block* DeviceCachingAllocator::try_allocate_expandable_block(int device, aclrtStream stream, BlockPool* pool, size_t size) {
    Block* candidate = find_expandable_block(device, stream, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped && !map_block(candidate, std::min(candidate->size, size))) {
        return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    while (candidate->size < size) {
        // invariant: free -> unmapped -> *
        // map_block will map some of unmapped and merge with free
        auto remaining = size - candidate->size;
        auto new_candidate = candidate->next;
        if (!map_block(new_candidate, std::min(remaining, candidate->next->size))) {
            return nullptr;
        }
        candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
}

void DeviceCachingAllocator::free_block(Block* block) {
    AT_ASSERT(!block->allocated && block->event_count == 0);

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
        const int64_t subsumed_size = static_cast<int64_t>(try_merge_blocks(block, merge_candidate, pool));
        if (subsumed_size > 0) {
            net_change_inactive_split_blocks -= 1;
            net_change_inactive_split_size -= subsumed_size;
        }
    }

    active_blocks.erase(block);
    pool.blocks.insert(block);

    if (block->is_split()) {
        net_change_inactive_split_blocks += 1;
        net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        // inactive_split tries to capture the idea that blocks
        // cannot be freed when requested, but fully free pages
        // of expandable blocks can always be freed.
        // The logic to track this as statistic is pretty involved,
        // so we simply just exclude expandable segements from
        // inactive_split
        if (!block->expandable_segment_) {
            update_stat(stats.inactive_split[stat_type], net_change_inactive_split_blocks);
            update_stat(stats.inactive_split_bytes[stat_type], net_change_inactive_split_size);
        }
        update_stat(stats.active[stat_type], -1);
        update_stat(stats.active_bytes[stat_type], -original_block_size);
        update_stat(stats.requested_bytes[stat_type], -static_cast<std::int64_t>(requested_size));
    });
}

size_t DeviceCachingAllocator::try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 || !src->stream_uses.empty() || dst->mapped != src->mapped) {
        return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {
        dst->ptr = src->ptr;
        dst->prev = src->prev;
        if (dst->prev) {
            dst->prev->next = dst;
        }
    } else {
        dst->next = src->next;
        if (dst->next) {
            dst->next->prev = dst;
        }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    delete src;
    src = nullptr;

    return subsumed_size;
}

DeviceCachingAllocator::LcPool& DeviceCachingAllocator::get_lc_pool(LifeCycleType lc) {
    switch (lc) {
        case LifeCycleType::DEFAULT_LC:
            return default_lc_pools;
        case LifeCycleType::LONG_LC:
        case LifeCycleType::FIRST_STEP_LC:
            return long_lc_pools;
    }
    AT_ASSERT(0);
}

BlockPool& DeviceCachingAllocator::get_pool(size_t size, LifeCycleType lc) {
    LcPool& pool = get_lc_pool(lc);
    if (size <= kSmallSize) {
        return pool.small_blocks;
    } else {
        return pool.large_blocks;
    }
}

std::vector<BlockPool*> DeviceCachingAllocator::get_pool_list(size_t size, LifeCycleType lc) {
    std::vector<LcPool*> pool_list = {&default_lc_pools, &long_lc_pools};
    LcPool& lc_pool = get_lc_pool(lc);
    int idx = 0;
    for (auto& x : pool_list) {
        if (x == &lc_pool) break;
        ++idx;
    }
    AT_ASSERT(idx < pool_list.size());
    std::vector<LcPool*> target_pool_list;
    for (int i = 0; i < (int)pool_list.size(); i++) {
        target_pool_list.push_back(pool_list[(idx + i) % pool_list.size()]);
    }
    std::vector<BlockPool*> target_list;
    for (auto& x : target_pool_list) {
        target_list.push_back(size <= kSmallSize ? &x->small_blocks : &x->large_blocks);
    }
    return target_list;
}

bool DeviceCachingAllocator::should_split(const Block* block, size_t size) {
    TORCH_INTERNAL_ASSERT(block->size >= size);
    size_t remaining = block->size - size;
    if (block->pool->is_small || CachingAllocatorConfig::expandable_segments()) {
        return remaining >= kMinBlockSize;
    } else {
        return (size < CachingAllocatorConfig::max_split_size()) && (remaining > kSmallSize);
    }
}

StatTypes DeviceCachingAllocator::get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
}

size_t DeviceCachingAllocator::get_allocation_size(size_t size, LifeCycleType lc) {
    if (lc == LifeCycleType::LONG_LC) {
        if (size <= kSmallSize) {
            return kSmallBuffer;
        } else {
            return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        }
    }
    if (size <= kSmallSize) {
        return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
        return kLargeBuffer;
    } else {
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

bool DeviceCachingAllocator::get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        // Track block reuse interval only when garbage collection is enabled.
        for (auto& b : pool.blocks) {
            ++b->gc_count;
        }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
        return false;
    }

    if ((*it)->expandable_segment_) {
        if (CachingAllocatorConfig::expandable_segments()) {
            // if we are allocated to the part of the block that is expandable
            // for the purposes of "best fit" we consider its size to be the size it
            // can expand to, not the size it currently is. This means that we
            // sometimes have to search for blocks with bigger 'size' before
            // choosing this segment.
            auto expandable_size = [](Block* b) { return b->size + (b->next && !b->next->mapped ? b->next->size : 0); };
            auto next = it;
            next++;
            while ((*it)->expandable_segment_ && next != pool.blocks.end() && (*next)->stream == p.stream() &&
                   expandable_size(*next) < expandable_size(*it)) {
                it = next++;
            }
        } else {
            // Rarely expandable segments has been turned off after we have
            // already allocated some blocks as expandable. For instance,
            // since we cannot share expandable memory via IPC, someone might
            // temporarily disable it. In this case we need to honor this request
            // by only finding non-expandable blocks
            do {
                it++;
            } while (it != pool.blocks.end() && (*it)->expandable_segment_ && (*it)->stream == p.stream());
            if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
                return false;
            }
        }
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
    // Short lifecycle tensor. In the short lifecycle memory pool, when in a non forward phase, 
    // to prevent large blocks from being occupied by small tensors and causing excessive fragmentation, myMaxSplitSize and
    // kLargeBuffer are used as restrictions.Overall, the purpose of this section is to prevent the 
    // generation of large fragments.
    if (!pool_idx && &pool == &default_lc_pools.large_blocks && (*it)->size >= myMaxSplitSize &&
        (*it)->size - p.size() >= kLargeBuffer && !_check()) {
        return false;
    }
    // Short lifecycle tensor, not allowed to be placed in the long lifecycle memory pool in the forward phase
    // In the non forward stage, when a long lifecycle block is idle,
    // it can store a short lifecycle tensor to improve memory reuse rate.
    if (&pool == &long_lc_pools.large_blocks && pool_idx && _check()) return false;
    // Long lifecycle tensor can only be placed in the long lifecycle memory pool and must achieve zero fragmentation.
    if (&pool == &long_lc_pools.large_blocks && !pool_idx) {
        if (p.alloc_size != (*it)->size || (*it)->prev || (*it)->next) {
            return false;
        }
    }
    // Long lifecycle tensor, not allowed to be placed in short lifecycle memory pool
    if (&pool == &default_lc_pools.large_blocks && pool_idx) {
        return false;
    }
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
}

bool DeviceCachingAllocator::get_free_block_memory_optimize(AllocParams &p, size_t tensor_forward_end,
                                                            size_t tensor_step_end, size_t tensor_forward_start,
                                                            size_t tensor_step_start) {
    BlockPool& pool = *p.pool;
    if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        // Track block reuse interval only when garbage collection is enabled.
        for (auto& b : pool.blocks) {
            ++b->gc_count;
        }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    bool flag = false;

    for (int i = 1; i <= DeviceCachingAllocator::prevent_memory_conflict_num; i++) {
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) return false;
        // Do not return an oversized block for a large request
        if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
            ((*it)->size >= CachingAllocatorConfig::max_split_size()))
            return false;
        // Allow oversized block size to be rounded up but within a limit
        if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer))
            return false;
        if (!pool_idx && &pool == &default_lc_pools.large_blocks && (*it)->size >= myMaxSplitSize &&
            (*it)->size - p.size() >= kLargeBuffer && !_check()) {
            return false;
        }
        if (&pool == &long_lc_pools.large_blocks && pool_idx && _check()) return false;
        if (&pool == &long_lc_pools.large_blocks && !pool_idx) {
            if (p.alloc_size != (*it)->size || (*it)->prev || (*it)->next) {
                return false;
            }
        }
        if (&pool == &default_lc_pools.large_blocks && pool_idx) {
            return false;
        }

        // Add up the allocs before and after the block
        size_t seg_size = (*it)->size;
        Block *befor_block = (*it)->prev;
        Block *next_block = (*it)->next;
        // look up from the front
        while (befor_block) {
            seg_size += befor_block->size;
            befor_block = befor_block->prev;
        }
        // look up from the end
        while (next_block) {
            seg_size += next_block->size;
            next_block = next_block->next;
        }

        // p.size() --> round_size、(*it)->orig_size时origin大小、(*it)->size时alloc大小
        if (seg_size >= p.size() + kSizeLimit) {
            // Determine if there are size blocks in the step within the lifecycle of the tensor.
            // If there are, it indicates that there will be block sized tensors generated within the lifecycle of the tensor
            bool is_tensor_in_step = malloc_recorder.has_tensor_in_step(tensor_step_start, tensor_step_end, seg_size);
            if (is_tensor_in_step) {
                // Find the last block of the iterator
                while (it != pool.blocks.end()) {
                    it++;
                    if (it == pool.blocks.end()) {
                        return false;
                    }
                    if ((*it)->device == p.search_key.device  && (*it)->stream == p.search_key.stream &&
                        p.search_key.size <= (*it)->size) {
                        break;
                    }
                }
                continue;
            } else {
                flag = true;
                break;
            }
        }
    }
    if (flag) {
        p.block = *it;
        pool.blocks.erase(it);
        return true;
    } else {
        return false;
    }
}

bool DeviceCachingAllocator::get_free_block_after_alloc(AllocParams &p) {
    BlockPool& pool = *p.pool;
    if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        // Track block reuse interval only when garbage collection is enabled.
        for (auto& b : pool.blocks) {
            ++b->gc_count;
        }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
        return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer)) return false;

    // Forward stage, short lifecycle tensor, cannot be placed in long lifecycle memory pool to prevent tensor conflicts
    if (&pool == &long_lc_pools.large_blocks) {
        if (pool_idx == 0 && _check()) return false;
    }
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
}

bool DeviceCachingAllocator::trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    return freed_memory;
}

void DeviceCachingAllocator::garbage_collect_cached_blocks() {
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
    const BlockPool* pools[] = {&long_lc_pools.large_blocks, &default_lc_pools.large_blocks};
    for (auto pool : pools) {
        for (auto& b : pool->blocks) {
            if (!b->is_split()) {
                total_age += b->gc_count;
                ++freeable_block_count;
            }
        }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
        return;
    }

    TORCH_CHECK(c10_npu::npuSynchronizeDevice(true), "npuSynchronizeDevice failed.");

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed && freeable_block_count > 0) {
        // Free blocks exceeding this age threshold first.
        double age_threshold = total_age / freeable_block_count;
        // Stop iteration if we can no longer free a block.
        block_freed = false;

        // Free blocks of > avg age. Don't stop upon reaching the target_size,
        // we don't want this GC to be triggered frequently.
        for (auto pool : pools) {
            auto it = pool->blocks.begin();
            while (it != pool->blocks.end()) {
                Block* block = *it;
                ++it;
                if (!block->is_split() && block->gc_count >= age_threshold) {
                    block_freed = true;
                    gc_reclaimed += block->size;
                    total_age -= block->gc_count; // Decrement the age
                    freeable_block_count--; // One less block that can be freed
                    release_block(block);

                    ASCEND_LOGD("PTA CachingAllocator gc: free = %zu, cached = %lu, allocated = %lu", block->size,
                                stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
                                stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
                }
            }
        }
    }
}

bool DeviceCachingAllocator::alloc_block(AllocParams& p, bool isRetry) {
    size_t size = p.alloc_size;
    void* ptr = nullptr;

    // In order to prevent the failure of aclrtMalloc_wrapper from consuming a lot of time, prediction is made in advance
    static size_t usable_total = 0;
    if (usable_total &&
        alr_total_size + size + DeviceCachingAllocator::memory_fail_prejudgment > usable_total && !isRetry) {
        return false;
    }

    if (isRetry) {
        stats.num_alloc_retries += 1;
    }

    if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
        p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
    } else if (CachingAllocatorConfig::expandable_segments()) {
        p.block = try_allocate_expandable_block(p.device(), p.stream(), p.pool, p.size());
        if (p.block) {
            p.err = ACL_ERROR_NONE;
        } else {
            p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
        }
        return bool(p.block);
    } else {
        p.err = aclrtMalloc_wrapper(&ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
        size_t device_free;
        size_t device_total;
        TORCH_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total) == ACL_ERROR_NONE, \
                                    "Error, failed to get memory info");
        TORCH_INTERNAL_ASSERT(device_free <= device_total);
        usable_total = alr_total_size + device_free;
        ASCEND_LOGD("pytorch-change code, reserved:%lu, free:%lu, reserved+free:%lu (after aclrtmalloc)\n", alr_total_size,
               device_free, alr_total_size + device_free);
    }

    if (p.err != ACL_ERROR_NONE) {
        return false;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
        update_stat(stats.segment[stat_type], 1);
        update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size()) update_stat(stats.oversize_segments, 1);
    ASCEND_LOGD("pta_memory acl_malloc: malloc = %zu, ret = %d", size, p.err);

    return (p.block != nullptr);
}

bool DeviceCachingAllocator::release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max()) {
        return false;
    }
    BlockPool &pool = *p.pool;
    Block key = p.search_key;
    key.size =
            (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
    auto it = pool.blocks.lower_bound(&key);

    TORCH_CHECK(c10_npu::npuSynchronizeDevice(true), "npuSynchronizeDevice failed.");

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

bool DeviceCachingAllocator::release_cached_blocks(bool check_error) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(check_error);

    // Free all non-split cached blocks
    TORCH_CHECK(c10_npu::npuSynchronizeDevice(check_error), "npuSynchronizeDevice failed.");
    release_blocks(long_lc_pools.large_blocks);
    release_blocks(long_lc_pools.small_blocks);
    release_blocks(default_lc_pools.large_blocks);
    release_blocks(default_lc_pools.small_blocks);

    return true;
}

void DeviceCachingAllocator::release_expandable_segment(Block* block) {
    TORCH_INTERNAL_ASSERT(block->size == block->expandable_segment_->size(), "block disagrees with segment");
    TORCH_INTERNAL_ASSERT(!block->mapped);
    auto it = std::find(expandable_segments_.begin(), expandable_segments_.end(), block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end());
    expandable_segments_.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment_;
    block->expandable_segment_ = nullptr;
    delete block;
    block = nullptr;
}

bool DeviceCachingAllocator::release_cached_blocks_long(bool check_error) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(check_error);

    // Free all non-split cached blocks
    TORCH_CHECK(c10_npu::npuSynchronizeDevice(check_error), "npuSynchronizeDevice failed.");
    release_blocks(long_lc_pools.large_blocks);
    release_blocks(long_lc_pools.small_blocks);

    return true;
}

bool DeviceCachingAllocator::release_cached_blocks_default(bool check_error) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(check_error);

    // Free all non-split cached blocks
    TORCH_CHECK(c10_npu::npuSynchronizeDevice(check_error), "npuSynchronizeDevice failed.");
    release_blocks(default_lc_pools.large_blocks);
    release_blocks(default_lc_pools.small_blocks);

    return true;
}

void DeviceCachingAllocator::release_block(Block* block) {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    TORCH_CHECK(aclrtFree_wrapper((void*)block->ptr) == ACL_ERROR_NONE, "aclrtFree_wrapper failed.");
    total_allocated_memory -= block->size;

    auto* pool = block->pool;

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.segment[stat_type], -1);
        update_stat(stats.reserved_bytes[stat_type], -block->size);
    });

    if (block->size >= CachingAllocatorConfig::max_split_size()) update_stat(stats.oversize_segments, -1);

    ASCEND_LOGD("pta_memory acl_free: free_size = %zu", block->size);

    pool->blocks.erase(block);
    delete block;
    block = nullptr;
}

void DeviceCachingAllocator::unmap_block(Block* block) {
    auto unmapped = block->expandable_segment_->unmap(SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
        return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size = static_cast<char*>(unmapped.ptr) - static_cast<char*>(block->ptr);
    if (before_size > 0) {
        // prev? -> before_free -> block
        Block* before_free = new Block(block->device, block->stream, before_size, block->pool, block->ptr);
        before_free->expandable_segment_ = block->expandable_segment_;
        before_free->splice(block->prev, block);
        block->pool->blocks.insert(before_free);
    }

    TORCH_CHECK(block->size >= before_size + unmapped.size, "after size should be greater than or equal to 0");
    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
        // block -> after_free -> next?
        Block* after_free = new Block(block->device, block->stream, after_size, block->pool,
                static_cast<char*>(unmapped.ptr) + unmapped.size);
        after_free->expandable_segment_ = block->expandable_segment_;
        after_free->splice(block, block->next);
        block->pool->blocks.insert(after_free);
    }

    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    TORCH_CHECK(try_merge_blocks(block, block->prev, *block->pool) >= 0, "try_merge_blocks failed.");
    TORCH_CHECK(try_merge_blocks(block, block->next, *block->pool) >= 0, "try_merge_blocks failed.");
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types,
                                [&](size_t stat_type) { update_stat(stats.reserved_bytes[stat_type], -unmapped.size); });
}

void DeviceCachingAllocator::release_blocks(BlockPool& pool) {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
        Block *block = *it;
        ++it;
        if (block->expandable_segment_) {
            // unmapping will mutate the free pool
            // so just gather what needs to be freed
            // to avoid invalidating the iterator
            to_unmap.push_back(block);
        } else if (!block->prev && !block->next) {
            release_block(block);
        }
    }
    for (Block* block : to_unmap) {
        unmap_block(block);
        if (!block->prev && !block->next) {
            release_expandable_segment(block);
        }
    }
}

EventPool::Event DeviceCachingAllocator::create_event_internal(int idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
}

void DeviceCachingAllocator::synchronize_and_free_events(bool check_error) {
    // Synchronize on outstanding events and then free associated blocks.
    for (auto& st : npu_events) {
        for (auto& e : st.second) {
            EventPool::Event event = std::move(e.first);
            Block* block = e.second;

            if (check_error) {
                TORCH_CHECK(aclrtSynchronizeEvent(*event) == ACL_ERROR_NONE, "acl interface call failed");
            } else {
                TORCH_CHECK(aclrtSynchronizeEvent(*event) == ACL_ERROR_NONE, "acl interface call failed");
            }
            ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed.");

            block->event_count--;
            if (block->event_count == 0) {
                free_block(block);
            }
        }
    }

    npu_events.clear();
}

void DeviceCachingAllocator::insert_events(Block* block) {
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
        TORCH_CHECK(c10_npu::SetDevice(stream.device_index()) == ACL_ERROR_NONE, "SetDevice failed.");

        EventPool::Event event = create_event_internal(stream.device_index());
        event->record(stream);
        ASCEND_LOGI("Event: record DeviceAllocator is successfully executed.");

        block->event_count++;
        npu_events[stream].emplace_back(std::move(event), block);
    }
    if (ret_ctx == ACL_ERROR_NONE) {
        TORCH_CHECK(aclrtSetCurrentContext(compiler_ctx) == ACL_ERROR_NONE, "aclrtSetCurrentContext failed.");
    }
}

void DeviceCachingAllocator::process_events() {
    // Process outstanding npuEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    for (auto it = npu_events.begin(); it != npu_events.end();) {
        while (!it->second.empty()) {
            auto& e = it->second.front();
            EventPool::Event event = std::move(e.first);
            Block* block = e.second;

            if (!event->query()) {
                e.first = std::move(event);
                break;
            }

            block->event_count--;
            if (block->event_count == 0) {
                free_block(block);
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

void DeviceCachingAllocator::cache_info_aux(BlockPool& blocks, size_t* total, size_t* largest) {
    for (auto it = blocks.blocks.begin(); it != blocks.blocks.end(); ++it) {
        size_t blocksize = (*it)->size;
        *total += blocksize;
        if (blocksize > *largest) {
            *largest = blocksize;
        }
    }
}