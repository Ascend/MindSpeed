# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.tensor_parallel.comm_autograd_function import auto_grad_scatter_along_last_dim
from mindspeed.core.tensor_parallel.comm_autograd_function import \
    auto_grad_sync_gather_along_last_dim
from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
from mindspeed.core.transformer.moe.tp_2d.grouped_mlp_2d import GroupedMLP2D
from mindspeed.core.transformer.moe.tp_2d.moe_allgather_token_dispatcher_2d import \
    MoEAllGatherTokenDispatcher2D
from mindspeed.core.transformer.moe.tp_2d.sequential_mlp_2d import SequentialMLP2D
from mindspeed.core.transformer.moe.tp_2d.topk_router_2d import TopKRouter2D


class MoELayer2D(BaseMoELayer):
    """2D Mixture of experts Layer **currently only supports allgather gmm**.

    """

    def __init__(
            self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        super(MoELayer2D, self).__init__(config=config, layer_number=layer_number)
        self.submodules = submodules
        self.router = TopKRouter2D(config=self.config)
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP2D(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP2D(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher2D(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, hidden_states: torch.Tensor):
        # [s/x, b, h/y] -> [s/x, b, h]
        hidden_states = auto_grad_sync_gather_along_last_dim(hidden_states, TPYCollectiveComm)

        #  [sb/x, h] => [sb/x, topK], [sb/x, topK]
        topk_probs, topk_indices = self.router(hidden_states)

        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(hidden_states, topk_probs,
                                                                                        topk_indices)
        expert_output, bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, bias)

        # [s/x, b, h] -> [s/x, b, h/y]
        output = auto_grad_scatter_along_last_dim(output, TPYCollectiveComm)
        if mlp_bias:
            mlp_bias = auto_grad_scatter_along_last_dim(mlp_bias, TPYCollectiveComm)

        return output, mlp_bias
