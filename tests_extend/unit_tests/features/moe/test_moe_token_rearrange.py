import pytest
import torch

from unit_tests.common import DistributedTest
from commons import set_random_seed

from mindspeed import megatron_adaptor
from mindspeed.moe.config import Config
from mindspeed.moe.gate import TopKGate
from mindspeed.moe.experts import Experts
from mindspeed.moe.moe_layer import MOELayer
from megatron.legacy.model.transformer import ParallelMLP
from megatron.core.transformer import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.parallel_state import get_expert_model_parallel_group, destroy_model_parallel, initialize_model_parallel


class TestMOELayer(DistributedTest):
    world_size = 4
    topk_gate = {
        "ne_4_k_1": TopKGate(Config(hidden_size=2, num_experts=4, topk=1)),
        "ne_4_k_2": TopKGate(Config(hidden_size=2, num_experts=4, topk=2)),
        "ne_2_k_2": TopKGate(Config(hidden_size=2, num_experts=2, topk=2)),
    }
    parallel_mlp = None

    def get_moe_layer_output(self, topk, input_data, ep_size, num_experts):
        expert = Experts(self.parallel_mlp, num_experts).npu()
        gate = self.topk_gate.get(f"ne_{num_experts}_k_{topk}", None)
        moe_layer_module = MOELayer(
            gate,
            expert,
            ep_size=ep_size,
            num_local_experts=num_experts // ep_size
        ).npu()
        expert_parallel_group = get_expert_model_parallel_group()
        moe_layer_module.set_ep_group(expert_parallel_group)

        return moe_layer_module(input_data)
    
    @pytest.mark.parametrize("tp_pp_ep", [(1, 1, 2)])
    @pytest.mark.parametrize("ne_topk", [(4, 1), (4, 2), (2, 2)])
    def test_moe_layer(self, tp_pp_ep, ne_topk):
        set_random_seed(1)
        (tp, pp, ep) = tp_pp_ep
        (num_experts, topk) = ne_topk
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=tp,
                                  pipeline_model_parallel_size=pp,
                                  virtual_pipeline_model_parallel_size=None,
                                  pipeline_model_parallel_split_rank=None,
                                  expert_model_parallel_size=ep)
        input_data = torch.randn(4, 2).npu()
        # base without token rearranged

        args_base = parse_args(None, True)
        args_base.enable_token_rearrange_opt = False
        args_base.moe_router_topk = topk
        set_args(args_base)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=2,
            num_attention_heads=4,
            use_cpu_initialization=True,
            fp16=True,
        )
        self.parallel_mlp = ParallelMLP(config)
        output_base = self.get_moe_layer_output(args_base.moe_router_topk, input_data, ep, num_experts)

        # with token rearraged
        args_rearrange = parse_args(None, True)
        args_rearrange.enable_token_rearrange_opt = False
        args_rearrange.moe_router_topk = topk
        set_args(args_rearrange)
        output_rearrage = self.get_moe_layer_output(args_rearrange.moe_router_topk, input_data, ep, num_experts)

        assert torch.allclose(output_base, output_rearrage, rtol=0.0001, atol=0.0001)
