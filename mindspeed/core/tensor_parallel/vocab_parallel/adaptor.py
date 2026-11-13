# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region
)
from mindspeed.core.tensor_parallel.vocab_parallel.vocab_parallel import (
    vocab_parallel_embedding_forward_impl
)


def mindspeed_vocab_parallel_embedding_forward(self, input_):
    return vocab_parallel_embedding_forward_impl(
        self,
        input_,
        reduce_from_tensor_model_parallel_region=reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region=reduce_scatter_to_sequence_parallel_region
    )
