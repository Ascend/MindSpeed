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

#include <mutex>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include "SwapException.h"

using CUmemGenericAllocationHandle = unsigned long long;

constexpr size_t granularitySize = 2097152;

struct Block;

struct BlockSegment {
    BlockSegment() : block(nullptr), offset(0) {}
    BlockSegment(Block *block, size_t offset) : block(block), offset(offset) {}

    Block *block;
    size_t offset;
};

class GMLakeError : public std::exception {
public:
    explicit GMLakeError(const std::string &message) : message(message) {}
    const char *what() const noexcept override
    {
        return message.c_str();
    }

private:
    std::string message;
};

struct PhyBlock {
    explicit PhyBlock(int device_id_in = -1, size_t block_size_in = granularitySize)
        : device_id(device_id_in),
          block_size(block_size_in),
          status(ACL_SUCCESS),
          free(true),
          owner_stream(nullptr),
          released(false)
    {
        if (device_id == -1) {
            SWAP_CHECK_ERROR(c10_npu::GetDevice(&device_id));
        }

        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.reserve = 0;
        status = aclrtMallocPhysical(&alloc_handle, block_size, &prop, 0);
        if (status != ACL_ERROR_NONE) {
            throw GMLakeError("PhyBlock Construct Failed");
        }
    }

    void release_resources()
    {
        if (status == ACL_SUCCESS) {
            auto err = aclrtFreePhysical(alloc_handle);
            if (err != ACL_SUCCESS) {
                throw GMLakeError("PhyBlock Release_resources Failed");
            }
            alloc_handle = nullptr;
        }
        released = true;
    }

    ~PhyBlock()
    {
        if (!released) {
            this->release_resources();
            released = true;
        }
    }

    int device_id;
    const size_t block_size;
    aclrtDrvMemHandle alloc_handle = nullptr;
    aclError status;

    bool free;
    aclrtStream owner_stream;
    std::vector<BlockSegment> mapped_blocks;
    bool released;
};

struct VirDevPtr {
    VirDevPtr(void *addr_in, size_t allocSize_in, int device_id = -1)
        : allocSize(allocSize_in), mapped(false), device_id(device_id), status(ACL_SUCCESS), released(false)
    {
        if (device_id == -1) {
            SWAP_CHECK_ERROR(c10_npu::GetDevice(&device_id));
        }

        void *device_ptr;
        void *request_ptr = addr_in;
        auto status = aclrtReserveMemAddress(&device_ptr, allocSize, 0, request_ptr, 1);
        if (status != ACL_SUCCESS || request_ptr != nullptr && device_ptr != request_ptr) {
            if (device_ptr != nullptr) {
                SWAP_CHECK_ERROR(aclrtReleaseMemAddress(device_ptr));
            }
            virAddr = nullptr;
            if (status == ACL_SUCCESS) {
                status = ACL_ERROR_FAILURE;
            }
            return;
        }

        virAddr = device_ptr;
    }

    void release_resources()
    {
        if (virAddr) {
            if (mapped) {
                for (size_t i = 0; i * granularitySize < allocSize; i++) {
                    SWAP_CHECK_ERROR(aclrtUnmapMem(virAddr + i * granularitySize));
                }
            }
            SWAP_CHECK_ERROR(aclrtReleaseMemAddress(virAddr));
        }

        released = true;
    }

    ~VirDevPtr()
    {
        if (!released) {
            this->release_resources();
            released = true;
        }
    }

    void *virAddr;
    const size_t allocSize;
    bool mapped;
    int device_id;
    aclError status;
    bool released;
};

struct VirBlock {
    VirBlock(std::shared_ptr<VirDevPtr> vir_dev_ptr_in, size_t offset_in, size_t blockSize_in,
        std::shared_ptr<PhyBlock> phy_block_in, int device_id = -1)
        : vir_dev_ptr(vir_dev_ptr_in),
          offset(offset_in),
          blockSize(blockSize_in),
          phy_block(phy_block_in),
          device_id(device_id),
          status(ACL_SUCCESS),
          released(false)
    {
        if (device_id == -1) {
            SWAP_CHECK_ERROR(c10_npu::GetDevice(&device_id));
        }

        block_ptr = (void *)(((char *)vir_dev_ptr->virAddr) + offset);
        void *device_ptr = block_ptr;
        SWAP_CHECK_ERROR(aclrtMapMem(device_ptr, blockSize, 0, phy_block->alloc_handle, 0));
        if (offset == 0) {
            vir_dev_ptr->mapped = true;
        }
    }

    void release_resources()
    {
        vir_dev_ptr.reset();
        released = true;
    }

    ~VirBlock()
    {
        if (!released) {
            this->release_resources();
            released = true;
        }
    }

    std::shared_ptr<VirDevPtr> vir_dev_ptr;

    size_t offset;
    size_t blockSize;
    void *block_ptr;

    std::shared_ptr<PhyBlock> phy_block;

    int device_id;
    aclError status;
    bool released;
};

struct VmmSegment {
    VmmSegment() : granul_size(0), segment_ptr(nullptr), status(ACL_SUCCESS), num_free_chunks(0), released(false) {}

    explicit VmmSegment(size_t blocks, size_t block_size_in = granularitySize, int device_id_in = -1)
        : granul_size(block_size_in),
          segment_ptr(nullptr),
          device_id(device_id_in),
          status(ACL_SUCCESS),
          num_free_chunks(blocks),
          num_used_chunks(0),
          fused(false),
          released(false)
    {
        if (device_id == -1) {
            SWAP_CHECK_ERROR(c10_npu::GetDevice(&device_id));
        }

        allocate_phy_chunks(blocks, block_size_in, device_id);
        if (status == ACL_SUCCESS) {
            mapVirAddr();
        }
    }

    explicit VmmSegment(std::vector<std::shared_ptr<PhyBlock>> &&phy_chunks_in)
        : phy_chunks(std::move(phy_chunks_in)),
          granul_size(phy_chunks[0]->block_size),
          segment_ptr(nullptr),
          device_id(phy_chunks[0]->device_id),
          status(ACL_SUCCESS),
          num_free_chunks(phy_chunks.size()),
          num_used_chunks(0),
          fused(true),
          released(false)
    {
        mapVirAddr();
    }

    explicit VmmSegment(std::vector<std::shared_ptr<PhyBlock>> phy_chunks_in,
        std::vector<std::shared_ptr<VirBlock>> vir_chunks_in)
        : phy_chunks(std::move(phy_chunks_in)),
          vir_chunks(std::move(vir_chunks_in)),
          granul_size(phy_chunks[0]->block_size),
          segment_ptr(vir_chunks[0]->block_ptr),
          device_id(phy_chunks[0]->device_id),
          status(ACL_SUCCESS),
          num_free_chunks(phy_chunks.size()),
          num_used_chunks(0),
          fused(false),
          released(false)
    {}

    void allocate_phy_chunks(size_t blocks, size_t block_size_in, int device_id_in)
    {
        phy_chunks.reserve(blocks);
        for (size_t i = 0; i < blocks; i++) {
            auto phy_block = std::make_shared<PhyBlock>(device_id_in, block_size_in);
            if (phy_block->status != ACL_SUCCESS) {
                size_t device_free;
                size_t device_total;

                status = phy_block->status;
                phy_chunks.clear();
                break;
            } else {
                phy_chunks.emplace_back(std::move(phy_block));
            }
        }
    }

    void release_resources()
    {
        {
            auto tmp_vir = std::move(vir_chunks);
        }
        {
            auto tmp_phy = std::move(phy_chunks);
        }
        released = true;
    }

    virtual ~VmmSegment()
    {
        if (!released) {
            this->release_resources();
            released = true;
        }
    }

    void *mapVirAddr()
    {
        static constexpr int retry_times = 8;
        static std::mutex alloc_mutex;

        void *device_ptr = nullptr;
        size_t segment_size = phy_chunks.size() * granul_size;

        int current_try = 0;
        aclError result = ACL_ERROR_NONE;
        do {
            std::lock_guard<std::mutex> lock(alloc_mutex);

            auto vir_dev_ptr = std::make_shared<VirDevPtr>(device_ptr, segment_size, device_id);
            device_ptr = vir_dev_ptr->virAddr;

            if (vir_dev_ptr->status != ACL_SUCCESS || !vir_dev_ptr->virAddr) {
                result = vir_dev_ptr->status;
            } else {
                vir_chunks.clear();

                size_t offset = 0;
                for (size_t j = 0; j < phy_chunks.size(); j++) {
                    auto phy_block = phy_chunks[j];
                    auto vir_block = std::make_shared<VirBlock>(vir_dev_ptr, offset, granul_size, phy_block, device_id);

                    if (vir_block->status != ACL_SUCCESS) {
                        result = vir_block->status;
                        vir_chunks.clear();
                        break;
                    } else {
                        vir_chunks.emplace_back(std::move(vir_block));
                    }

                    offset += granul_size;
                }
            }

            current_try++;
            device_ptr = nullptr;
        } while (result != ACL_SUCCESS && current_try < retry_times);

        status = result;
        if (result == ACL_ERROR_NONE) {
            segment_ptr = vir_chunks[0]->block_ptr;
            return segment_ptr;
        }

        return nullptr;
    }

    std::shared_ptr<VmmSegment> split(size_t keep_size)
    {
        size_t keep_blocks = keep_size / granul_size;

        std::vector<std::shared_ptr<PhyBlock>> remain_phy_chunks;
        std::vector<std::shared_ptr<VirBlock>> remain_vir_chunks;

        size_t remaining_free_blocks = 0;
        for (size_t i = keep_blocks; i < phy_chunks.size(); i++) {
            if (phy_chunks[i]->free) {
                remaining_free_blocks++;
            }
            remain_phy_chunks.emplace_back(std::move(phy_chunks[i]));
            remain_vir_chunks.emplace_back(std::move(vir_chunks[i]));
        }

        this->phy_chunks.resize(keep_blocks);
        this->vir_chunks.resize(keep_blocks);

        auto remaining_segment =
            std::make_shared<VmmSegment>(std::move(remain_phy_chunks), std::move(remain_vir_chunks));

        remaining_segment->segment_ptr = (void *)((char *)segment_ptr + keep_size);
        remaining_segment->num_free_chunks = remaining_free_blocks;

        num_free_chunks -= remaining_free_blocks;
        return remaining_segment;
    }

    bool remerge(VmmSegment &segment)
    {
        if (segment.segment_ptr ==
            static_cast<void *>(static_cast<char *>(this->segment_ptr) + this->phy_chunks.size() * granul_size)) {
            for (size_t i = 0; i < segment.phy_chunks.size(); i++) {
                this->phy_chunks.emplace_back(std::move(segment.phy_chunks[i]));
                this->vir_chunks.emplace_back(std::move(segment.vir_chunks[i]));
            }
        } else if (this->segment_ptr ==
            static_cast<void *>(static_cast<char *>(segment.segment_ptr) + segment.phy_chunks.size() * granul_size)) {
            for (size_t i = 0; i < phy_chunks.size(); i++) {
                segment.phy_chunks.emplace_back(std::move(this->phy_chunks[i]));
                segment.vir_chunks.emplace_back(std::move(this->vir_chunks[i]));
            }

            this->phy_chunks = std::move(segment.phy_chunks);
            this->vir_chunks = std::move(segment.vir_chunks);

            this->segment_ptr = segment.segment_ptr;
        } else {
            throw GMLakeError("remerge(VmmSegment& segment)");
            return false;
        }

        this->num_free_chunks += segment.num_free_chunks;
        segment.num_free_chunks = 0;

        segment.phy_chunks.clear();
        segment.vir_chunks.clear();

        segment.segment_ptr = nullptr;

        return true;
    }

    std::vector<std::shared_ptr<PhyBlock>> phy_chunks;
    std::vector<std::shared_ptr<VirBlock>> vir_chunks;

    const size_t granul_size;
    void *segment_ptr;

    int device_id;
    aclError status;

    size_t num_free_chunks;
    size_t num_used_chunks;
    bool fused;
    bool released;
};
