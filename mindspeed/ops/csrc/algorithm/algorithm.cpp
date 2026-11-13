/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

void reuse_data_ptr(at::Tensor& des, at::Tensor& src, int64_t offset)
{
    TORCH_CHECK(
        offset >= 0,
        "Expect offset equal or greater than zero, got: ", offset);

    TORCH_CHECK(
        (offset + des.numel()) * des.element_size() <=
        src.numel() * src.element_size(),
        "Offsets overflow, got: ",
        "offset ", offset * des.element_size(),
        ", des storage size ", des.numel() * des.element_size(),
        ", src storage size ", src.numel()* src.element_size());

    char* data_ptr = static_cast<char*>(src.storage().data_ptr().get()) + offset * des.element_size();
    at::DataPtr aim_data_ptr = at::DataPtr(data_ptr, des.storage().device());
    des.storage().set_data_ptr(std::move(aim_data_ptr));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reuse_data_ptr", &reuse_data_ptr, "reuse tensor data ptr");
}