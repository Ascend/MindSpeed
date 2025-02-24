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

#include <limits>
#include <vector>
#include <memory>

#include <c10/util/flat_hash_map.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include "NPUVmmApi.h"

using c10_npu::NPUCachingAllocator::DeviceStats;
using c10_npu::NPUCachingAllocator::RecordContext;
using c10_npu::NPUCachingAllocator::SegmentInfo;
using c10_npu::NPUCachingAllocator::Stat;
using c10_npu::NPUCachingAllocator::StatArray;
using c10_npu::NPUCachingAllocator::StatType;
using c10_npu::NPUCachingAllocator::TraceEntry;
// using c10_npu::NPUCachingAllocator::History;
using OutOfMemoryObserver =
    std::function<void(int64_t device, int64_t allocated, int64_t device_total, int64_t device_free)>;

struct History {
    void *addr;
    size_t real_size;                              // unrounded, actually requested size
    std::shared_ptr<c10::GatheredContext> context; // per-watcher context
};

struct BlockInfo {
    int64_t size = 0;
    int64_t requested_size = 0;
    int32_t gc_counter = 0;
    bool allocated = false;
    bool active = false;
    std::shared_ptr<c10::GatheredContext> context_when_allocated;
    std::vector<History> history;
};

using stream_set = ska::flat_hash_set<c10_npu::NPUStream>;

using CreateContextFn = std::shared_ptr<c10::GatheredContext> (*)(void);

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocs to 2 MiB
constexpr size_t kGranularity = 2097152;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat &stat, int64_t amount);

void reset_accumulated_stat(Stat &stat);

void reset_peak_stat(Stat &stat);

template <typename Func> void for_each_selected_stat_type(const StatTypes &stat_types, Func f)
{
    for (const auto stat_type : c10::irange(stat_types.size())) {
        if (stat_types[stat_type]) {
            f(stat_type);
        }
    }
}

void update_stat_array(StatArray &stat_array, int64_t amount, const StatTypes &stat_types);

struct Block;
using Comparison = bool (*)(const Block *, const Block *);

struct BlockPool {
    BlockPool(Comparison comparator, bool small) : blocks(comparator), is_small(small) {}
    std::set<Block *, Comparison> blocks;
    std::unordered_set<size_t> hash;
    const bool is_small;
};

struct HistoryChain {
    History h;
    std::unique_ptr<HistoryChain> next; // when blocks are merged we keep records
                                        // of what used to be in the block
};

struct Block {
    int device;             // gpu
    aclrtStream stream;     // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size;            // block size in bytes
    size_t requested_size;  // memory originally requested
    size_t actual_size;
    BlockPool *pool{ nullptr }; // owning memory pool
    void *ptr{ nullptr };       // memory address
    bool allocated{ false };    // in-use flag
    Block *prev{ nullptr };     // prev block if split from a larger allocation
    Block *next{ nullptr };     // next block if split from a larger allocation
    int event_count{ 0 };       // number of outstanding CUDA events
    int gc_count{ 0 };          // counter for prioritizing older / less useful blocks for
                                // garbage collection
    std::unique_ptr<HistoryChain> history{ nullptr };
    HistoryChain *history_last{ nullptr };
    std::shared_ptr<VmmSegment> vmm_segment;
    size_t ptr_hash;
    // std::shared_ptr<BlockEvent> self_last_event;

    Block(int device, aclrtStream stream, size_t size, BlockPool *pool, void *ptr)
        : device(device),
          stream(stream),
          stream_uses(),
          size(size),
          actual_size(0),
          requested_size(0),
          pool(pool),
          // self_last_event(std::make_shared<BlockEvent>(stream)),
          ptr(ptr)
    {
        ptr_hash = reinterpret_cast<size_t>(ptr);
    }

    // constructor for search key
    Block(int device, aclrtStream stream, size_t size)
        : device(device),
          stream(stream),
          stream_uses(),
          size(size),
          actual_size(0),
          // self_last_event(std::make_shared<BlockEvent>(stream)),
          requested_size(0)
    {
        ptr_hash = 0;
    }

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }

    void splice(Block *before, Block *after)
    {
        if (before) {
            before->next = this;
        }
        prev = before;
        if (after) {
            after->prev = this;
        }
        next = after;
    }
};

struct BlockHash {
    size_t operator () (const Block *b) const
    {
        return b->ptr_hash;
    }
};

bool BlockComparator(const Block *a, const Block *b);

using EventOrderedBlockSet = std::unordered_set<Block *, BlockHash>;
using SetIterator = EventOrderedBlockSet::iterator;

struct BlockEventOrderPool {
    BlockEventOrderPool() : pool_size(0) {}

    void insert(Block *block)
    {
        if (blocks.count(block) == 0) {
            blocks.insert(block);
            pool_size += block->size;
        }
    }

    bool erase(Block *block)
    {
        if (blocks.count(block)) {
            blocks.erase(block);
            pool_size -= block->size;

            return true;
        } else {
            return false;
        }
    }

    SetIterator erase(SetIterator it)
    {
        if (blocks.count(*it)) {
            pool_size -= (*it)->size;

            return blocks.erase(it);
        } else {
            return blocks.end();
        }
    }

    EventOrderedBlockSet blocks;
    size_t pool_size;
};

inline std::string format_size(uint64_t size)
{
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
    AllocParams(int device, size_t size, aclrtStream stream, BlockPool *pool, size_t alloc_size, DeviceStats &stats)
        : search_key(device, stream, size), pool(pool), alloc_size(alloc_size), block(nullptr), err(ACL_ERROR_NONE)
    {}

    int device() const
    {
        return search_key.device;
    }
    aclrtStream stream() const
    {
        return search_key.stream;
    }
    size_t size() const
    {
        return search_key.size;
    }

    Block search_key;
    BlockPool *pool;
    size_t alloc_size;
    Block *block;
    StatTypes stat_types = { false };
    aclError err;
};
