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
#include "EventPool.h"

EventPool::Event EventPool::get(int device)
{
    auto &pool = pools_[device];
    auto destructor = [&pool](c10_npu::NPUEvent *event) {
        std::lock_guard<std::mutex> g(pool.mutex_);
        pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
        std::lock_guard<std::mutex> g(pool.mutex_);
        if (!pool.event_pool_.empty()) {
            auto *event = pool.event_pool_.back().release();
            pool.event_pool_.pop_back();
            return Event(event, destructor);
        }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS).release(), destructor);
}

void EventPool::empty_cache()
{
    for (auto &pool : pools_) {
        std::lock_guard<std::mutex> g(pool.mutex_);
        pool.event_pool_.clear();
    }
}
