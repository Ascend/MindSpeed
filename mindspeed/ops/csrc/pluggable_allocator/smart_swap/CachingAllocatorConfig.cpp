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
#include "CachingAllocatorConfig.h"

#include "common.h"

void CachingAllocatorConfig::lexArgs(const char *env, std::vector<std::string> &config)
{
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

void CachingAllocatorConfig::consumeToken(const std::vector<std::string> &config, size_t i, const char c) {}

size_t CachingAllocatorConfig::parseMaxSplitSize(const std::vector<std::string> &config, size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        size_t val1 = static_cast<size_t>(stoi(config[i]));
        val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
        val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
        m_max_split_size = val1 * 1024 * 1024;
    }
    return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i)
{
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
        double val1 = stod(config[i]);
        m_garbage_collection_threshold = val1;
    }
    return i;
}

void CachingAllocatorConfig::parseArgs(const char *env)
{
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
        }

        if (i + 1 < config.size()) {
            consumeToken(config, ++i, ',');
        }
    }
}
