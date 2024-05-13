#ifndef PLUGGABLEALLOCATOR_EVENTPOOL_H
#define PLUGGABLEALLOCATOR_EVENTPOOL_H

#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include "common.h"

class EventPool {
public:
    using Event = std::unique_ptr<c10_npu::NPUEvent, std::function<void(c10_npu::NPUEvent*)>>;
    // Explicit device count
    EventPool() : pools_(c10_npu::device_count()) {}

    Event get(int device);

    void empty_cache();

private:
    struct PerDevicePool {
        alignas(64) std::mutex mutex_;
        std::vector<std::unique_ptr<c10_npu::NPUEvent>> event_pool_;
    };
    std::vector<PerDevicePool> pools_;
};

#endif
