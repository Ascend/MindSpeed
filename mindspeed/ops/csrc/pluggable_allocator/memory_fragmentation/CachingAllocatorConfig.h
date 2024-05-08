#ifndef PLUGGABLEALLOCATOR_CACHINGALLOCATORCONFIG_H
#define PLUGGABLEALLOCATOR_CACHINGALLOCATORCONFIG_H

#include "common.h"
#include "Recorder.h"

class CachingAllocatorConfig {
public:
    static size_t max_split_size();

    static double garbage_collection_threshold();

    static bool expandable_segments();

    static double default_lc_threshold();

    static bool open_memory_optimize();

    static CachingAllocatorConfig &instance();

    void parseArgs(const char* env);

private:
    size_t m_max_split_size;
    double m_garbage_collection_threshold;
    bool m_expandable_segments;
    double m_default_lc_threshold;
    bool m_open_memory_optimize;

    CachingAllocatorConfig();

    void lexArgs(const char* env, std::vector<std::string>& config);
    void consumeToken(const std::vector<std::string>& config, size_t i, const char c);
    size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
    size_t parseGarbageCollectionThreshold(const std::vector<std::string>& config, size_t i);
    size_t parseExpandableSegments(const std::vector<std::string>& config, size_t i);
    size_t parseDefaultLcThreshold(const std::vector<std::string>& config, size_t i);
    size_t parseOpenMemoryOptimize(const std::vector<std::string>& config, size_t i);
};

#endif
