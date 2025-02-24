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
#pragma once

#include <limits>
#include <vector>
#include <string>

class CachingAllocatorConfig {
public:
    static size_t max_split_size()
    {
        return instance().m_max_split_size;
    }

    static double garbage_collection_threshold()
    {
        return instance().m_garbage_collection_threshold;
    }

    static CachingAllocatorConfig &instance()
    {
        static CachingAllocatorConfig *s_instance = ([]() {
            auto inst = new CachingAllocatorConfig();
            const char *env = getenv("PYTORCH_NPU_ALLOC_CONF");
            inst->parseArgs(env);
            return inst;
        })();
        return *s_instance;
    }

    void parseArgs(const char *env);

private:
    size_t m_max_split_size;
    double m_garbage_collection_threshold;

    CachingAllocatorConfig()
        : m_max_split_size(std::numeric_limits<size_t>::max()),
          m_garbage_collection_threshold(0) {}

    void lexArgs(const char *env, std::vector<std::string> &config);
    void consumeToken(const std::vector<std::string> &config, size_t i, const char c);
    size_t parseMaxSplitSize(const std::vector<std::string> &config, size_t i);
    size_t parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i);
};
