#include "common.h"


std::mutex alr_lock;
std::unordered_map<void *, size_t> alr_recorder;
size_t alr_total_size = 0;
size_t alr_get_max() {
    static size_t rc = 0;
    if (rc == 0) {
        char *e = getenv("ALR_MAX");
        if (e) {
            std::string s(e);
            long long alr_max = 0;
            try {
                alr_max = std::stoll(s);
            } catch (const std::invalid_argument& e) {
                TORCH_CHECK(false, "Error, expecting digital string in ALR_MAX");
            } catch (const std::out_of_range& e) {
                TORCH_CHECK(false, "Error, out of long long range");
            }
            if (alr_max < 0) {
                TORCH_CHECK(false, "Error,  alr_max cannot be nagative number");
            } else {
                rc = static_cast<size_t>(alr_max);
            }
        } else {
            rc = LLONG_MAX;
        }
        printf("ALR MAX SET  %lu\n", rc);
    }
    return rc;
}


aclError aclrtMalloc_wrapper(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
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


aclError aclrtFree_wrapper(void *devPtr) {
    alr_lock.lock();
    TORCH_INTERNAL_ASSERT(alr_total_size >= alr_recorder[devPtr]);
    alr_total_size -= alr_recorder[devPtr];
    alr_recorder.erase(devPtr);
    alr_lock.unlock();
    return aclrtFree(devPtr);
}


void update_stat(Stat &stat, int64_t amount) {
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }
    if (amount < 0) {
        stat.freed += -amount;
    }
}


void reset_accumulated_stat(Stat& stat) {
    stat.allocated = 0;
    stat.freed = 0;
}


void reset_peak_stat(Stat& stat) { stat.peak = stat.current; }


void update_stat_array(StatArray& stat_array, int64_t amount, const StatTypes& stat_types) {
    for_each_selected_stat_type(stat_types,
                                [&stat_array, amount](size_t stat_type) { update_stat(stat_array[stat_type], amount); });
}


std::string get_block_pool_str(BlockPoolType type) {
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


bool BlockComparatorSize(const Block* a, const Block* b) {
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

bool BlockComparatorAddress(const Block* a, const Block* b) {
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

std::string format_size(uint64_t size) {
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
