#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <vector>

#include "NpuCachingCustomAllocator.h"

std::mutex *NpuCachingCustomAllocator::getFreeMutex() const {
  static std::mutex npu_free_mutex;
  return &npu_free_mutex;
}

Block *NpuCachingCustomAllocator::get_allocated_block(void *ptr, bool remove) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = allocated_blocks.find(ptr);
  if (it == allocated_blocks.end()) {
    return nullptr;
  }
  Block *block = it->second;
  if (remove) {
    allocated_blocks.erase(it);
  }
  return block;
}

void NpuCachingCustomAllocator::init(int device_count) {
  int size = static_cast<int>(device_allocator.size());
  if (size < device_count) {
    device_allocator.resize(device_count);
    for (const auto i : c10::irange(size, device_count)) {
      device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
    }
  }
}

bool NpuCachingCustomAllocator::initialized() { return !device_allocator.empty(); }

/** allocates a block which is safe to use from the provided stream */
void *NpuCachingCustomAllocator::malloc(int device, size_t size, aclrtStream stream) {
  Block *block = device_allocator[device]->malloc(device, size, stream);
  add_allocated_block(block);
  void *devPtr = static_cast<void *>(block->ptr);
  return devPtr;
}

void NpuCachingCustomAllocator::free(void *ptr) {
  if (!ptr) {
    return;
  }
  Block *block = get_allocated_block(ptr, true);
  if (!block) {
    AT_ERROR("invalid device pointer: ", ptr);
  }
  device_allocator[block->device]->free(block);
}

void NpuCachingCustomAllocator::emptyCache(bool check_error) {
  int count = static_cast<int>(device_allocator.size());
  for (int i = 0; i < count; i++) device_allocator[i]->emptyCache(check_error);
}

void NpuCachingCustomAllocator::assertValidDevice(int device) {
  int device_num = c10_npu::device_count();
  AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats NpuCachingCustomAllocator::getDeviceStats(int device) {
  assertValidDevice(device);
  return device_allocator[device]->getStats();
}

void NpuCachingCustomAllocator::resetPeakStats(int device) {
  assertValidDevice(device);
  device_allocator[device]->resetPeakStats();
}

std::string NpuCachingCustomAllocator::name() { return "native"; }

void CachingAllocatorConfig::lexArgs(const char *env, std::vector<std::string> &config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void CachingAllocatorConfig::consumeToken(const std::vector<std::string> &config, size_t i, const char c) {
  TORCH_CHECK(i < config.size() && config[i].compare(std::string(1, c)) == 0,
              "Error parsing CachingAllocator settings, expected ", c);
}

size_t CachingAllocatorConfig::parseMaxSplitSize(const std::vector<std::string> &config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = static_cast<size_t>(stoi(config[i]));
    TORCH_CHECK(val1 > kLargeBuffer / (1024 * 1024), "CachingAllocator option max_split_size_mb too small, must be > ",
                kLargeBuffer / (1024 * 1024));
    val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value");
  }
  return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0");
    TORCH_CHECK(val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(false, "Error, expecting garbage_collection_threshold value");
  }
  return i;
}

size_t CachingAllocatorConfig::parseExpandableSegments(const std::vector<std::string> &config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(i < config.size() && (config[i] == "True" || config[i] == "False"),
                "Expected a single True/False argument for expandable_segments");
    m_expandable_segments = (config[i] == "True");
    if (m_expandable_segments) {
      void *ptr = nullptr;
      auto status = aclrtReserveMemAddress(&ptr, 512, 0, NULL, 1);
      if (status == ACL_ERROR_NONE) {
        aclrtReleaseMemAddress(ptr);
      } else {
        NPU_CHECK_SUPPORTED_OR_ERROR(status);
        m_expandable_segments = false;
      }
    }
  } else {
    TORCH_CHECK(false, "Error, expecting expandable_segments value");
  }
  return i;
}

void CachingAllocatorConfig::parseArgs(const char *env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_garbage_collection_threshold = 0;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i].compare("max_split_size_mb") == 0) {
      i = parseMaxSplitSize(config, i);
    } else if (config[i].compare("garbage_collection_threshold") == 0) {
      i = parseGarbageCollectionThreshold(config, i);
    } else if (config[i] == "expandable_segments") {
      i = parseExpandableSegments(config, i);
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }
}

NpuCachingCustomAllocator my_allocator;
void local_raw_delete(void *ptr) { my_allocator.free(ptr); }