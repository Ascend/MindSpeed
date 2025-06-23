# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import torch


def allreduce_layernorm_grads(model, config):
    """allreduce layernorm grads"""

    from megatron.core import parallel_state
    from megatron.core.utils import get_attr_wrapped_model
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
            config.sequence_parallel or config.qk_layernorm
    ):
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if (
                        getattr(param, 'sequence_parallel', False) or "_norm" in name
                        or 'q_layernorm' in name
                        or 'k_layernorm' in name
                ):
                    grad = param.main_grad
                    torch.distributed.all_reduce(
                        grad, group=parallel_state.get_tensor_model_parallel_group()
                    )
