import pytest
import torch
import torch_npu
from ascendspeed import megatron_adaptor

from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.global_vars import set_args
from megatron.arguments import parse_args


class TestRecompute(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    set_args(args)

    @pytest.mark.parametrize("granularity", ['selective', 'full'])
    @pytest.mark.parametrize("method", ['uniform', 'block'])
    @pytest.mark.parametrize("distribute_saved_activations", [True, False])
    def test_checkpoint_recompute(self, granularity, method, distribute_saved_activations):
        config_tuple = (granularity, method, distribute_saved_activations)
        self.checkpoint_recompute(config_tuple)

    def test_checkpoint_recompute_memory(self):
        config_list = []
        config_list.append(('None', 'None', False))
        config_list.append(('selective', 'None', False))
        config_list.append(('full', 'block', False))

        memory_allocated = []
        # get allocated memory of each recomputing config
        for config in config_list:
            memory_allocated.append(self.checkpoint_recompute(config))

        # allocated memory, None > selective > full
        assert memory_allocated[0] > memory_allocated[1] > memory_allocated[2]

    def checkpoint_recompute(self, config_tuple):
        granularity, method, distribute_saved_activations = config_tuple
        initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(312)

        config = TransformerConfig(num_layers=4, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        config.recompute_granularity = granularity
        config.recompute_method = method
        config.recompute_num_layers = 4
        config.distribute_saved_activations = distribute_saved_activations
        transformer_block = TransformerBlock(config, get_gpt_layer_local_spec(), post_layer_norm=False)
        assert transformer_block.config.recompute_granularity == granularity
        assert transformer_block.config.recompute_method == method
        assert transformer_block.checkpoint_core_attention if granularity == 'selective' \
            else ~transformer_block.checkpoint_core_attention

        sequence_length = 32
        micro_batch_size = 2
        transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = transformer_block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size
        return torch.cuda.memory_allocated()
