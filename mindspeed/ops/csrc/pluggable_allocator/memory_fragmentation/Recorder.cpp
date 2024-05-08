#include <iostream>
#include "Recorder.h"

unsigned int pool_idx = 0;
bool g_record_flag = false;
size_t g_record_cnt = 0;
bool is_precise_match = false;
std::mutex MemoryRecorder::lock;    // protect the static value
std::unordered_set<MemoryRecorder *> MemoryRecorder::recorder_set;

bool MallocRecorder::in_step_flag;
std::mutex MallocRecorder::lock;
std::unordered_set<MallocRecorder *> MallocRecorder::recorder_set; // Recorder_set records MallocRecorder objects

void set_g_record_flag(bool flag) {
    g_record_flag = flag;
}

void set_is_precise_match(bool flag) {
    is_precise_match = flag;
}

bool MemoryRecorder::_RecorderComparator(const RecEle a, const RecEle b) {
    if (a.tensor_forward_start_tik != b.tensor_forward_start_tik) {
        return a.tensor_forward_start_tik < b.tensor_forward_start_tik;
    } else if (a.tensor_forward_end_tik != b.tensor_forward_end_tik) {
        return a.tensor_forward_end_tik < b.tensor_forward_end_tik;
    } else
        return a.size < b.size;
}

MemoryRecorder::MemoryRecorder()
    : lc_id_cnt(0), forward_tik(1), forward_count(0), forward_rec_set(_RecorderComparator), last_forward_rec_set(_RecorderComparator) {
     lock.lock();
     recorder_set.insert(this);
     lock.unlock();
}

MemoryRecorder::~MemoryRecorder() {
    lock.lock();
    recorder_set.erase(this);
    lock.unlock();
}

void MemoryRecorder::add(size_t tensor_forward_start_tik, size_t tensor_forward_end_tik, size_t origin_size) {
    if (_check()) {
        forward_rec_set.emplace(tensor_forward_start_tik, tensor_forward_end_tik, origin_size);
    }
}

LifeCycleType MemoryRecorder::get_lc(size_t origin_size, size_t *tensor_forward_end, size_t *tensor_forward_start) {
    // Mark all tensors in the initialization phase of the optimizer as long lifecycle
    if (is_precise_match) {
        return LifeCycleType::LONG_LC;
    }
    // All tensors that are not in the forward stage are marked as short lifecycles
    if (!_check()) {
        return LifeCycleType::DEFAULT_LC;
    }
    // All tensors from setup_model to the end of the first forward stage are marked as long lifecycle
    if(last_forward_rec_set.size() == 0) {
        return LifeCycleType::LONG_LC;
    }

    long lc_id = lc_id_cnt++;
    // Find tensor in all branches
    // Find in current branch
    auto it1 = last_forward_rec_set.upper_bound(
            RecEle(forward_tik, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
    while (it1 != last_forward_rec_set.begin()) {
        it1--;
        if (it1->size == origin_size && it1->tensor_forward_end_tik == std::numeric_limits<size_t>::max()) {
            *tensor_forward_start = it1->tensor_forward_start_tik;
            *tensor_forward_end = it1->tensor_forward_end_tik;
            return LifeCycleType::LONG_LC;
        } else if (it1->size == origin_size) {
            // Record the forward start and end periods of the current tensor
            *tensor_forward_start = it1->tensor_forward_start_tik;
            *tensor_forward_end = it1->tensor_forward_end_tik;
            return LifeCycleType::DEFAULT_LC;
        }
    }

    // Search in other branches
    for (auto &other_branch : forward_history) {
        it1 = other_branch.upper_bound(
                RecEle(forward_tik, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
        while (it1 != other_branch.begin()) {
            it1--;
            if (it1->size == origin_size && it1->tensor_forward_end_tik == std::numeric_limits<size_t>::max()) {
                *tensor_forward_start = it1->tensor_forward_start_tik;
                *tensor_forward_end = it1->tensor_forward_end_tik;
                return LifeCycleType::LONG_LC;
            } else if (it1->size == origin_size) {
                *tensor_forward_start = it1->tensor_forward_start_tik;
                *tensor_forward_end = it1->tensor_forward_end_tik;
                return LifeCycleType::DEFAULT_LC;
            }
        }
    }
    // If it cannot be found in all branches and the number of tensors in the current branch is greater than lc_id,
    // it indicates that it may be a new branch.
    if (last_forward_rec_set.size() >= lc_id) {
        has_Another_Situation = true;
    }

    // If it cannot be found in all branches, it indicates that it may be a new branch
    return LifeCycleType::DEFAULT_LC;
}


void MemoryRecorder::change_forward_end_tik(size_t start_tik, size_t end_tik, size_t origin_size, unsigned int forward_distance, bool in_forward) {
    if (in_forward && _check() && forward_distance == 0) {
        auto it = forward_rec_set.find(RecEle(start_tik, std::numeric_limits<size_t>::max(), origin_size));
        if (it != forward_rec_set.end()) {
            forward_rec_set.erase(it);
            forward_rec_set.emplace(RecEle(start_tik, end_tik, origin_size));
        }
    }
}

void MemoryRecorder::start_record() {
    if (!is_precise_match && g_record_cnt++) {
        g_record_flag = true;
    }
    for (auto i:recorder_set) {
        i->lock.unlock();
        i->forward_tik = 1;
    }
}

void MemoryRecorder::_end_record() {
    lc_id_cnt = 0;
    forward_tik = 1;
    forward_count++;
    // If there is a fork, store the tensor record of the previous forward in forward_history
    if (forward_count == 0 || has_Another_Situation) {
        forward_history.emplace_back(forward_rec_set);
        has_Another_Situation = false;
    }
    last_forward_rec_set = forward_rec_set;
    forward_rec_set.clear();
}

void MemoryRecorder::end_record() {
    g_record_flag = false;
    lock.lock();
    for (auto i:recorder_set) {
        i->_end_record();
    }
    lock.unlock();
}


bool MallocRecorder::_RecorderComparator(const MallocRecorderEle a, const MallocRecorderEle b) {
    if (a.start_tik != b.start_tik) {
        return a.start_tik < b.start_tik;
    } else if (a.end_tik != b.end_tik) {
        return a.end_tik < b.end_tik;
    } else
        return a.size < b.size;
}

bool MallocRecorder::_check() {
    return in_step_flag;
}

void MallocRecorder::add(size_t start_tik, size_t end_tik, size_t round_size) {
    if (MallocRecorder::_check()) {
        rec_set.emplace(MallocRecorderEle(start_tik, end_tik, round_size));
    }
}

void MallocRecorder::change_end_tik(size_t start_tik, size_t end_tik, size_t round_size,size_t step_distance, bool in_step) {
    if (in_step && MallocRecorder::_check() && step_distance == 0) {
        auto it = rec_set.find(MallocRecorderEle(start_tik, std::numeric_limits<size_t>::max(), round_size));
        if (it != rec_set.end()) {
            rec_set.erase(it);
            rec_set.emplace(start_tik, end_tik, round_size);
        }
    }
}

bool MallocRecorder::predict_long(size_t round_size, size_t *tensor_step_end, size_t *tensor_step_start) {
    if (!MallocRecorder::_check()) return false;
    long step_lc_id = step_lc_id_cnt++;
    // Find in the current branch
    auto it1 = last_rec_set.upper_bound(
            MallocRecorderEle(tik, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
    while (it1 != last_rec_set.begin()) {
        it1--;
        if (it1->size == round_size && it1->end_tik == std::numeric_limits<size_t>::max()) {
            *tensor_step_start = it1->start_tik;
            *tensor_step_end = it1->end_tik;
            return true;
        } else if (it1->size == round_size) {
            // Record the start and end times of the tensor in the step phase
            *tensor_step_start = it1->start_tik;
            *tensor_step_end = it1->end_tik;
            return false;
        }
    }
    // Search in the history branch
    for (auto &other_branch : step_history) {
        it1 = other_branch.upper_bound(
                MallocRecorderEle(tik, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
        while (it1 != other_branch.begin()) {
            it1--;
            if (it1->size == round_size && it1->end_tik == std::numeric_limits<size_t>::max()) {
                *tensor_step_start = it1->start_tik;
                *tensor_step_end = it1->end_tik;
                return true;
            } else if (it1->size == round_size) {
                *tensor_step_start = it1->start_tik;
                *tensor_step_end = it1->end_tik;
                return false;
            }
        }
    }
    // If it cannot be found in all branches and the number of tensors in the current branch is greater than step_lc_id,
    // it indicates that it may be a new branch
    if (last_rec_set.size() >= step_lc_id) {
        has_Another_Situation = true;
    }
    return false;
}

bool MallocRecorder::has_tensor_in_step(size_t tensor_step_start, size_t tensor_step_end, size_t seg_size) {
    // Find in the current branch
    auto it1 = last_rec_set.upper_bound(
            MallocRecorderEle(tensor_step_start, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
    while (it1 != last_rec_set.end()) {
        size_t alloc_size = MallocRecorder::get_allocation_size(it1->size);
        if (it1->end_tik == std::numeric_limits<size_t>::max()) {
            alloc_size = MallocRecorder::get_allocation_size(it1->size, LifeCycleType::LONG_LC);
        }

        if (it1->start_tik < tensor_step_end && alloc_size == seg_size) {
            return true;
        }
        it1++;
    }

    // Search in the history branch
    for (auto &other_branch : step_history) {
        it1 = other_branch.upper_bound(
                MallocRecorderEle(tensor_step_start, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
        while (it1 != other_branch.end()) {
            size_t alloc_size = MallocRecorder::get_allocation_size(it1->size);
            if (it1->end_tik == std::numeric_limits<size_t>::max()) {
                alloc_size = MallocRecorder::get_allocation_size(it1->size, LifeCycleType::LONG_LC);
            }

            if (it1->start_tik < tensor_step_end && alloc_size == seg_size) {
                return true;
            }
            it1++;
        }
    }
    return false;
}

size_t MallocRecorder::get_allocation_size(size_t size, LifeCycleType lc) {
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

MallocRecorder::MallocRecorder() : step_lc_id_cnt(0), tik(1), step_count(0), rec_set(_RecorderComparator), last_rec_set(_RecorderComparator) {
    lock.lock();
    recorder_set.insert(this);
    lock.unlock();
}

MallocRecorder::~MallocRecorder() {
   lock.lock();
   recorder_set.erase(this);
   lock.unlock();
}

void MallocRecorder::start_record() {
    in_step_flag = true;
    lock.lock();
    for (auto i : recorder_set) {
        i->lock.unlock();
        i->tik = 1;
    }
}

void MallocRecorder::end_record() {
    in_step_flag = false;
    lock.lock();
    for (auto i : recorder_set) {
        i->lock.unlock();
        i->step_lc_id_cnt = 0;
        i->tik = 1;
        i->step_count++;
        // If there is a fork, store the tensor record of the previous forward in forward_history
        if (i->step_count == 0 || i->has_Another_Situation) {
            i->step_history.emplace_back(i->rec_set);
            i->has_Another_Situation = false;
        }
        i->last_rec_set = i->rec_set;
        i->rec_set.clear();
    }
}