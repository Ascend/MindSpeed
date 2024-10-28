#include "EventPool.h"

EventPool::Event EventPool::get(int device) {
    TORCH_INTERNAL_ASSERT(0 <= device && static_cast<size_t>(device) < pools_.size(),
                            "device index out of range");
    auto& pool = pools_[device];
    auto destructor = [&pool](c10_npu::NPUEvent* event) {
        std::lock_guard<std::mutex> g(pool.mutex_);
        pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
        std::lock_guard<std::mutex> g(pool.mutex_);
        if (!pool.event_pool_.empty()) {
            auto* event = pool.event_pool_.back().release();
            pool.event_pool_.pop_back();
            return Event(event, destructor);
        }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS).release(), destructor);
}

void EventPool::empty_cache() {
    for (auto &pool : pools_) {
        std::lock_guard<std::mutex> g(pool.mutex_);
        pool.event_pool_.clear();
    }
}