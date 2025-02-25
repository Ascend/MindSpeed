# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


class SequentialMLP2D(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.
    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.add_bias = config.add_bias_linear
        self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):

        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output, output_bias = expert(hidden)

            output_local[start:end] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_local[start:end, :] = output_bias
        return output_local, output_bias_local

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Maps local expert to global experts."""
        if self.moe_extended_tp:
            raise NotImplementedError(
                'Currently distributed checkpointing is not supported for moe_extended_tp'
            )

        sharded_state_dict = {}
        num_global_experts = (
                parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        expert_sharded_prefix = f'{prefix}experts.'
        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                        len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(with_context_parallel=True),
                )

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
