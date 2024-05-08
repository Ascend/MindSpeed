#include <sys/types.h>
#include <iostream>
#include <torch/extension.h>

#include "acl_base.h"
#include "acl_rt.h"
#include "PluggableAllocator.h"
#include "Decorator.h"
#include "Recorder.h"

extern "C" {
void *memory_fragmentation_malloc(size_t size, int device, aclrtStream stream)
{
    void *ptr;
    ptr = PluggableAllocator::getInstance().malloc(device, size, stream);
    return ptr;
}

void memory_fragmentation_free(void *ptr, size_t size, int device, aclrtStream stream)
{
    PluggableAllocator::getInstance().free(ptr);
}

void memory_fragmentation_init(int device_count)
{
    PluggableAllocator::getInstance().init(device_count);
}

void memory_fragmentation_empty_cache(bool check_error)
{
    PluggableAllocator::getInstance().emptyCache(true);
}

void memory_fragmentation_memory_fraction(double fraction, int device)
{
    PluggableAllocator::getInstance().setMemoryFraction(fraction, device);
}

DeviceStats memory_fragmentation_get_device_stats(int device)
{
    return PluggableAllocator::getInstance().getDeviceStats(device);
}

void my_reset_peak_stats(int device)
{
    return PluggableAllocator::getInstance().resetPeakStats(device);
}
}

namespace memory_recorder_test {

using RecSet =
    std::set<MemoryRecorder::RecEle,
             bool (*)(MemoryRecorder::RecEle, MemoryRecorder::RecEle)>;
void add(RecSet &rec_set, size_t tensor_forward_start_tik,
         size_t tensor_forward_end_tik, size_t origin_size) {
  rec_set.emplace(MemoryRecorder::RecEle(tensor_forward_start_tik,
                                         tensor_forward_end_tik, origin_size));
}

bool test_setup_tensor_lc() {
  MemoryRecorder recorder;
  set_is_precise_match(false);
  set_g_record_flag(true);

  size_t forward_start, forward_end;
  // 从setup_model到第一个forward阶段结束的tensor全部标记为长生命周期
  return LifeCycleType::LONG_LC ==
         recorder.get_lc(0, &forward_end, &forward_start);
}

bool test_forward_tensor_long_lc() {
  MemoryRecorder recorder;
  set_is_precise_match(false);
  set_g_record_flag(true);

  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  size_t forward_start, forward_end;
  // 当前分支中查找
  return LifeCycleType::LONG_LC ==
         recorder.get_lc(2, &forward_end, &forward_start);
}

bool test_forward_tensor_short_lc() {
  MemoryRecorder recorder;
  set_is_precise_match(false);
  set_g_record_flag(true);

  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 4);
  add(recorder.last_forward_rec_set, 1, 9, 5);

  size_t forward_start, forward_end;
  // 当前分支中查找
  return LifeCycleType::DEFAULT_LC ==
         recorder.get_lc(5, &forward_end, &forward_start);
}

bool test_forward_other_branch_long_lc() {
  MemoryRecorder recorder;
  set_is_precise_match(false);
  set_g_record_flag(true);

  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  RecSet s(MemoryRecorder::_RecorderComparator);
  add(s, 1, std::numeric_limits<size_t>::max(), 5);
  add(s, 1, std::numeric_limits<size_t>::max(), 6);
  add(s, 1, std::numeric_limits<size_t>::max(), 7);
  add(s, 1, std::numeric_limits<size_t>::max(), 8);

  recorder.forward_history.push_back(s);
  size_t forward_start, forward_end;
  // 在其他分支中查找
  return LifeCycleType::LONG_LC ==
         recorder.get_lc(6, &forward_end, &forward_start);
}

bool test_forward_other_branch_short_lc() {
  MemoryRecorder recorder;
  set_is_precise_match(false);
  set_g_record_flag(true);

  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_forward_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  RecSet s(MemoryRecorder::_RecorderComparator);
  add(s, 1, std::numeric_limits<size_t>::max(), 5);
  add(s, 1, std::numeric_limits<size_t>::max(), 6);
  add(s, 1, 9, 7);
  add(s, 1, std::numeric_limits<size_t>::max(), 8);

  recorder.forward_history.push_back(s);
  size_t forward_start, forward_end;
  // 在其他分支中查找
  return LifeCycleType::DEFAULT_LC ==
         recorder.get_lc(7, &forward_end, &forward_start);
}

bool test_change_forward_end_tik() {
  MemoryRecorder recorder;
  set_g_record_flag(true);

  add(recorder.forward_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.forward_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.forward_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.forward_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  recorder.change_forward_end_tik(1, 100, 3, 0, true);
  return recorder.forward_rec_set.find(MemoryRecorder::RecEle(1, 100, 3)) != recorder.forward_rec_set.end();
}
}


namespace malloc_recorder_test {

using RecSet =
    std::set<MallocRecorder::MallocRecorderEle,
        bool (*)(MallocRecorder::MallocRecorderEle, MallocRecorder::MallocRecorderEle)>;
void add(RecSet &rec_set, size_t start_tik,
         size_t end_tik, size_t round_size) {
  rec_set.emplace(MallocRecorder::MallocRecorderEle(start_tik,
                                         end_tik, round_size));
}

bool test_step_tensor_long_lc() {
  MallocRecorder recorder;
  MallocRecorder::in_step_flag = true;

  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  size_t step_start, step_end;
  // find in current branch
  return recorder.predict_long(2, &step_end, &step_start);
}


bool test_step_tensor_short_lc() {
  MallocRecorder recorder;
  MallocRecorder::in_step_flag = true;

  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 4);
  add(recorder.last_rec_set, 1, 9, 5);

  size_t step_start, step_end;
  // find in current branch
  return !recorder.predict_long(5, &step_end, &step_start);
}


bool test_step_other_branch_long_lc() {
  MallocRecorder recorder;
  MallocRecorder::in_step_flag = true;

  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  RecSet s(MallocRecorder::_RecorderComparator);
  add(s, 1, std::numeric_limits<size_t>::max(), 5);
  add(s, 1, std::numeric_limits<size_t>::max(), 6);
  add(s, 1, std::numeric_limits<size_t>::max(), 7);
  add(s, 1, std::numeric_limits<size_t>::max(), 8);

  recorder.step_history.push_back(s);
  size_t step_start, step_end;
  // find in other branch
  return recorder.predict_long(6, &step_end, &step_start);
}


bool test_step_other_branch_short_lc() {
  MallocRecorder recorder;
  MallocRecorder::in_step_flag = true;

  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.last_rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  RecSet s(MallocRecorder::_RecorderComparator);
  add(s, 1, std::numeric_limits<size_t>::max(), 5);
  add(s, 1, std::numeric_limits<size_t>::max(), 6);
  add(s, 1, 9, 7);
  add(s, 1, std::numeric_limits<size_t>::max(), 8);

  recorder.step_history.push_back(s);
  size_t step_start, step_end;
  // find in other branch
  return !recorder.predict_long(7, &step_end, &step_start);
}

bool test_change_end_tik() {
  MallocRecorder recorder;
  MallocRecorder::in_step_flag = true;

  add(recorder.rec_set, 1, std::numeric_limits<size_t>::max(), 1);
  add(recorder.rec_set, 1, std::numeric_limits<size_t>::max(), 2);
  add(recorder.rec_set, 1, std::numeric_limits<size_t>::max(), 3);
  add(recorder.rec_set, 1, std::numeric_limits<size_t>::max(), 4);

  recorder.change_end_tik(1, 100, 3, 0, true);
  return recorder.rec_set.find(MallocRecorder::MallocRecorderEle(1, 100, 3)) != recorder.rec_set.end();
}

}

namespace device_caching_allocator_test {
#include "DeviceCachingAllocator.h"

bool test_round_size() {
  return DeviceCachingAllocator::round_size(200) == kMinBlockSize && DeviceCachingAllocator::round_size(1025) == 3 * kMinBlockSize;
}
}


namespace caching_allocator_config_test {
#include "CachingAllocatorConfig.h"

bool test_parse_args() {
  auto &config = CachingAllocatorConfig::instance();
  const char *args = "max_split_size_mb:40, garbage_collection_threshold:0.5, expandable_segments:False, "
                     "default_lc_threshold:128.5, open_memory_optimize:1";
  config.parseArgs(args);
  return config.max_split_size() == 40 * 1024 * 1024 && config.garbage_collection_threshold() == 0.5 &&
         !config.expandable_segments() && config.default_lc_threshold() == 128.5 && config.open_memory_optimize();
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("memory_recorder_start", &Decorator::memory_recorder_start, "start mark the life cycle of a tensor in forward");
    m.def("memory_recorder_end", &Decorator::memory_recorder_end, "end mark the life cycle of a tensor in forward");
    m.def("malloc_recorder_start", &Decorator::malloc_recorder_start, "start mark the life cycle of a tensor in step");
    m.def("malloc_recorder_end", &Decorator::malloc_recorder_end, "end mark the life cycle of a tensor in step");
    m.def("precise_match_start", &Decorator::precise_match_start, "start mark the life cycle of a tensor in optimizer init stage");
    m.def("precise_match_end", &Decorator::precise_match_end, "end mark the life cycle of a tensor in optimizer init stage");

    // 以下为ut用接口，非业务接口
    m.def("test_setup_tensor_lc", &memory_recorder_test::test_setup_tensor_lc, "");
    m.def("test_forward_tensor_long_lc", &memory_recorder_test::test_forward_tensor_long_lc, "");
    m.def("test_forward_tensor_short_lc", &memory_recorder_test::test_forward_tensor_short_lc, "");
    m.def("test_forward_other_branch_long_lc", &memory_recorder_test::test_forward_other_branch_long_lc, "");
    m.def("test_forward_other_branch_short_lc", &memory_recorder_test::test_forward_other_branch_short_lc, "");
    m.def("test_change_forward_end_tik", &memory_recorder_test::test_change_forward_end_tik, "");

    m.def("test_step_tensor_long_lc", &malloc_recorder_test::test_step_tensor_long_lc, "");
    m.def("test_step_tensor_short_lc", &malloc_recorder_test::test_step_tensor_short_lc, "");
    m.def("test_step_other_branch_long_lc", &malloc_recorder_test::test_step_other_branch_long_lc, "");
    m.def("test_step_other_branch_short_lc", &malloc_recorder_test::test_step_other_branch_short_lc, "");
    m.def("test_change_end_tik", &malloc_recorder_test::test_change_end_tik, "");

    m.def("test_round_size", &device_caching_allocator_test::test_round_size, "");

    m.def("test_parse_args", &caching_allocator_config_test::test_parse_args, "");
}