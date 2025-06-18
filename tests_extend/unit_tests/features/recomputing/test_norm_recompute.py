import types
import torch
from mindspeed import megatron_adaptor

from tests_extend.commons import set_random_seed, initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from mindspeed.core.memory.recompute.norm.adaptor import mindspeed_norm_recompute_forward
from mindspeed.model.transformer import set_attention_mask


class TestNormRecompute(DistributedTest):
    world_size = 8
    args = parse_args(None, True)
    set_args(args)

    def test_norm_recompute(self):
        args = parse_args(None, True)
        args.recompute_norm = True
        args.num_layers = 4
        args.recompute_norm_num_layers = 2
        args.transformer_pipeline_model_parallel_size = 2
        set_args(args)
        self.norm_recopute()

    def norm_recopute(self):
        initialize_model_parallel(2, 2)
        model_parallel_cuda_manual_seed(312)

        config = TransformerConfig(num_layers=4, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        config.hidden_dropout = 0
        config.attention_dropout = 0
        transformer_block_ref = TransformerBlock(config, get_gpt_layer_local_spec(), post_layer_norm=True)
        transformer_block_test = TransformerBlock(config, get_gpt_layer_local_spec(), post_layer_norm=True)
        transformer_block_test.load_state_dict(transformer_block_ref.state_dict().copy())

        for layer in transformer_block_test.layers:
            layer.forward = types.MethodType(mindspeed_norm_recompute_forward, layer)

        sequence_length = 32
        micro_batch_size = 2
        transformer_block_ref.cuda()
        transformer_block_test.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states_ref = torch.rand((sequence_length, micro_batch_size, config.hidden_size)).cuda()
        hidden_states_ref.requires_grad = True
        hidden_states_test = hidden_states_ref.clone().detach()
        hidden_states_test.requires_grad = True

        attention_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        set_attention_mask(attention_mask)

        out_ref = transformer_block_ref(hidden_states=hidden_states_ref, attention_mask=attention_mask)
        out_test = transformer_block_test(hidden_states=hidden_states_test, attention_mask=attention_mask)
        assert(torch.allclose(out_ref, out_test))

        out_ref.backward(torch.ones_like(out_ref))
        out_test.backward(torch.ones_like(out_ref))
        assert(torch.allclose(hidden_states_ref.grad, hidden_states_test.grad))
