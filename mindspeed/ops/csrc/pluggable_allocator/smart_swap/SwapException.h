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

#include <third_party/acl/inc/acl/acl_base.h>


inline const char *getSwapErrorFunction(const char *msg)
{
    return msg;
}

// If there is just 1 provided C-string argument, use it.
inline const char *getSwapErrorFunction(const char * /* msg */, const char *args)
{
    return args;
}

#define SWAP_CHECK_ERROR(err_code, ...)                                                                           \
    do {                                                                                                          \
        auto Error = err_code;                                                                                    \
        if ((Error) != ACL_ERROR_NONE) {                                                                          \
            TORCH_CHECK(false, __func__, ":", __FILE__, ":", __LINE__,                                            \
                " SWAP NPU function error: ", getSwapErrorFunction(#err_code, ##__VA_ARGS__), ", error code is ", \
                Error)                                                                                            \
        }                                                                                                         \
    } while (0)
