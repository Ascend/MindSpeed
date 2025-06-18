import types
import torch
from mindspeed import megatron_adaptor
from mindspeed.model.transformer import set_attention_mask
from mindspeed.features_manager.recompute.activation_function import RecomputeActivationFeature
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from tests_extend.commons import initialize_model_parallel
from tests_extend.unit_tests.common import DistributedTest


class ActRecomputeFeatureTset:
    @staticmethod
    def act_recompute_register_patch(patch_manager):
        args = parse_args(None, True)
        args.num_layers = 4
        args.recompute_activation_function = True
        args.recompute_activation_function_num_layers = 2
        set_args(args)

        recompute_activation_feature_func = RecomputeActivationFeature()
        recompute_activation_feature_func.validate_args(args)

        recompute_activation_feature_func.register_patches(patch_manager, args)
        patch_manager.apply_patches()

    @staticmethod
    def del_recompute_register_patch(patch_manager):
        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward', MLP.forward, force_patch=True)
        patch_manager.remove_wrappers('megatron.core.transformer.transformer_layer.TransformerLayer.__init__', 'parallel_transformer_layer_init_wrapper')
        patch_manager.apply_patches()


class TestActivationRecompute(DistributedTest):
    world_size = 8

    def test_activation_recompute(self):
        ActRecomputeFeatureTset().act_recompute_register_patch(pm)
        self.activation_recopute()
        ActRecomputeFeatureTset().del_recompute_register_patch(pm)

    def activation_recopute(self):
        initialize_model_parallel(2, 2)
        model_parallel_cuda_manual_seed(312)

        submodules = MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )

        config = TransformerConfig(num_layers=4, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        config.hidden_dropout = 0
        config.attention_dropout = 0
        config.layer_number = 4
        config.transformer_pipeline_model_parallel_size = 2
        transformer_block_ref = MLP(config, submodules=submodules)
        transformer_block_test = MLP(config, submodules=submodules)
        transformer_block_test.load_state_dict(transformer_block_ref.state_dict().copy())

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

        transformer_block_ref.layer_number = 4

        out_ref, _ = transformer_block_ref(hidden_states=hidden_states_ref)
        out_test, _ = transformer_block_test(hidden_states=hidden_states_test)

        assert(torch.allclose(out_ref, out_test))

        out_ref.backward(torch.ones_like(out_ref))
        out_test.backward(torch.ones_like(out_ref))
        assert(torch.allclose(hidden_states_ref.grad, hidden_states_test.grad))