import random
import torch
import numpy

import megatron.core.parallel_state as ps


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )
