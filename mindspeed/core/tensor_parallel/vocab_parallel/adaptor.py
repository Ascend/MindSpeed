# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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
