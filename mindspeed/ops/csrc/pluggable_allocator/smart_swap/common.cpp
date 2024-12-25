// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright 2022 The GLake Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "common.h"

void update_stat(Stat &stat, int64_t amount)
{
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0) {
        stat.allocated += amount;
    }
    if (amount < 0) {
        stat.freed += -amount;
    }
}

void reset_accumulated_stat(Stat &stat)
{
    stat.allocated = 0;
    stat.freed = 0;
}

void reset_peak_stat(Stat &stat)
{
    stat.peak = stat.current;
}

void update_stat_array(StatArray &stat_array, int64_t amount, const StatTypes &stat_types)
{
    for_each_selected_stat_type(stat_types,
        [&stat_array, amount](size_t stat_type) { update_stat(stat_array[stat_type], amount); });
}

bool BlockComparator(const Block *a, const Block *b)
{
    if (a->stream != b->stream) {
        return reinterpret_cast<uintptr_t>(a->stream) < reinterpret_cast<uintptr_t>(b->stream);
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}
