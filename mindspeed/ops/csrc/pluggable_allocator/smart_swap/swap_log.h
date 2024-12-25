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

#include <iostream>

enum class SwapLogLevel {
    SWAP_DEBUG = 0,
    SWAP_INFO = 1,
    SWAP_WARN = 2,
    SWAP_ERROR = 3,
    SWAP_NONE = 4,
};

class SwapLogApi {
public:
    static bool IsLogEnable(SwapLogLevel logLevel)
    {
        static bool INIT_SWAP_LOG_LEVEL = false;
        static int GLOBAL_SWAP_LOG_LEVEL = 0;
        if (!INIT_SWAP_LOG_LEVEL) {
            const char *levelStr = std::getenv("SWAP_LOG_LEVEL");
            int curLevel = static_cast<int>(SwapLogLevel::SWAP_ERROR);
            if (levelStr != nullptr) {
                int level = std::atoi(levelStr);
                if (level >= static_cast<int>(SwapLogLevel::SWAP_DEBUG) &&
                    level <= static_cast<int>(SwapLogLevel::SWAP_NONE)) {
                    curLevel = level;
                }
            }

            GLOBAL_SWAP_LOG_LEVEL = curLevel;
            INIT_SWAP_LOG_LEVEL = true;
        }
        return (GLOBAL_SWAP_LOG_LEVEL <= static_cast<int>(logLevel));
    }

    static int GetLogRank()
    {
        const char *envStr = std::getenv("RANK");
        int64_t envRank = (envStr != nullptr) ? strtol(envStr, nullptr, 10) : -1;
        return static_cast<int>(envRank);
    }

    static bool IsLogRankEnable(int rank)
    {
        static bool INIT_SWAP_LOG_RANK = false;
        static int GLOBAL_SWAP_LOG_RANK = -1;
        if (!INIT_SWAP_LOG_RANK) {
            const char *envStr = std::getenv("SWAP_LOG_RANK");
            int64_t envRank = (envStr != nullptr) ? strtol(envStr, nullptr, 10) : -1;
            int curRank = static_cast<int>(envRank);
            if (curRank >= -1 && curRank < 8) {
                GLOBAL_SWAP_LOG_RANK = curRank;
            }
            INIT_SWAP_LOG_RANK = true;
        }
        if (GLOBAL_SWAP_LOG_RANK == -1 || rank == -1 || GLOBAL_SWAP_LOG_RANK == rank) {
            return true;
        }
        return false;
    }
};

#define SWAP_LOG_DEBUG(fmt, ...)                                                             \
    do {                                                                                     \
        if (SwapLogApi::IsLogEnable(SwapLogLevel::SWAP_DEBUG)) {                             \
            const char * const funcName = __FUNCTION__;                                      \
            int rank = SwapLogApi::GetLogRank();                                             \
            if (SwapLogApi::IsLogRankEnable(rank)) {                                         \
                printf("[SWAP_DEBUG] %s:%d:%s,rank[%d]: " #fmt "\n", __FILENAME__, __LINE__, \
                    static_cast<const char *>(funcName), rank, ##__VA_ARGS__);               \
            }                                                                                \
        }                                                                                    \
    } while (false)

#define SWAP_LOG_INFO(fmt, ...)                                                             \
    do {                                                                                    \
        if (SwapLogApi::IsLogEnable(SwapLogLevel::SWAP_INFO)) {                             \
            const char * const funcName = __FUNCTION__;                                     \
            int rank = SwapLogApi::GetLogRank();                                            \
            if (SwapLogApi::IsLogRankEnable(rank)) {                                        \
                printf("[SWAP_INFO] %s:%d:%s,rank[%d]: " #fmt "\n", __FILENAME__, __LINE__, \
                    static_cast<const char *>(funcName), rank, ##__VA_ARGS__);              \
            }                                                                               \
        }                                                                                   \
    } while (false)

#define SWAP_LOG_WARN(fmt, ...)                                                             \
    do {                                                                                    \
        if (SwapLogApi::IsLogEnable(SwapLogLevel::SWAP_WARN)) {                             \
            const char * const funcName = __FUNCTION__;                                     \
            int rank = SwapLogApi::GetLogRank();                                            \
            if (SwapLogApi::IsLogRankEnable(rank)) {                                        \
                printf("[SWAP_WARN] %s:%d:%s,rank[%d]: " #fmt "\n", __FILENAME__, __LINE__, \
                    static_cast<const char *>(funcName), rank, ##__VA_ARGS__);              \
            }                                                                               \
        }                                                                                   \
    } while (false)

#define SWAP_LOG_ERROR(fmt, ...)                                                             \
    do {                                                                                     \
        if (SwapLogApi::IsLogEnable(SwapLogLevel::SWAP_ERROR)) {                             \
            const char * const funcName = __FUNCTION__;                                      \
            int rank = SwapLogApi::GetLogRank();                                             \
            if (SwapLogApi::IsLogRankEnable(rank)) {                                         \
                printf("[SWAP_ERROR] %s:%d:%s,rank[%d]: " #fmt "\n", __FILENAME__, __LINE__, \
                    static_cast<const char *>(funcName), rank, ##__VA_ARGS__);               \
            }                                                                                \
        }                                                                                    \
    } while (false)
